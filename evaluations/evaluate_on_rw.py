import argparse
import pickle
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar

import cv2
import numpy as np
import tensorflow as tf
from loess.loess_1d import loess_1d
from matplotlib import pyplot as plt
from tqdm import tqdm

from stereo.evaluations.evaluate_dispersion import get_dispersion
from stereo.evaluations.evaluate_forward_error import (
    _load_config,
    _maximum_3d_error,
    _mean_reprojection_error,
    _parse_methods,
)
from stereo.evaluations.utils import implementation_from_config
from stereo.implementations import IMPLEMENTATIONS
from stereo.implementations.custom_optimization_procedure import get_rays_batched, squared_reprojection_error

IMAGE_SHAPE = (1280, 720)

TypeT = TypeVar("TypeT", bound="FullScene")


class AbstractScene:
    @cached_property
    def sigma(self) -> Any:
        return np.sqrt(np.mean(self.errors**2))

    @cached_property
    def covariances(self) -> Any:
        return [
            np.cov(
                (
                    (self.ml_views[:, i, :] - self.gt_views[:, i, :])
                    / shortest_quad_side(self.gt_views)[:, None]
                ).T
            )
            for i in range(4)
        ]

    @property
    def rectangle(self) -> np.ndarray:
        if self._rectangle is None:
            implementation = IMPLEMENTATIONS["LM-rigid_body"]
            implementation.set_hyperparameters(steps=1000)
            self._rectangle = np.array(implementation(self.gt_views, self.cameras))
        return self._rectangle

    @cached_property
    def diversity(self) -> np.ndarray:
        return get_dispersion(
            (tf.constant(self.gt_views), tf.constant(self.cameras))
        ).numpy()

    @cached_property
    def corner_rays(self) -> np.ndarray:
        _, corner_rays = get_rays_batched(
            tf.constant(self.gt_views), tf.constant(self.cameras)
        )
        return corner_rays.numpy()

    @cached_property
    def number_of_views(self) -> int:
        return len(self.cameras)

    @cached_property
    def errors(self) -> np.ndarray:
        return (self.ml_views - self.gt_views) / shortest_quad_side(self.gt_views)[
            :, None, None
        ]

    @cached_property
    def t_squared(self) -> bool:
        err_means = np.mean(self.errors, axis=1)[..., None]
        result = []
        for err_mean, covariance in zip(err_means, self.covariances):
            result.append(
                self.number_of_views * err_mean.T @ np.linalg.inv(covariance) @ err_mean
            )
        return np.max(np.array(result)) <= 3.56  # note this is valid for k=20 only

    @cached_property
    def likelihood_ratio(self) -> bool:
        u = (4 * np.linalg.det(self.covariances)) / (
            np.trace(self.covariances, axis1=-1, axis2=-2)
        ) ** 2
        nu = self.number_of_views - 1
        u_prime = -(nu - 1) * np.log(u)
        return np.max(u_prime) < 0.103


@dataclass
class FullScene(AbstractScene):
    name: str
    images: list[np.ndarray]
    gt_views: np.ndarray
    ml_views: np.ndarray
    cameras: np.ndarray
    _rectangle: field(default=None)

    # pylint: disable=no-member
    @classmethod
    def from_path(cls: type[TypeT], path: Path) -> TypeT:
        gt_views = []
        ml_views = []
        cameras = []
        images = []
        for filename in sorted((path / "images").iterdir()):
            images.append(cv2.imread(str(filename)))
            with (path / "ML" / filename.with_suffix(".npy").name).open(
                "rb"
            ) as ml_file:
                ml_views.append(np.load(ml_file) * IMAGE_SHAPE)
            with (path / "GT" / filename.with_suffix(".npy").name).open(
                "rb"
            ) as gt_file:
                gt_views.append(np.load(gt_file) * IMAGE_SHAPE)
            with (path / "camera" / filename.with_suffix(".npy").name).open(
                "rb"
            ) as cam_file:
                cameras.append(np.load(cam_file))
        return cls(
            path.stem,
            images,
            np.array(gt_views, np.float32),
            np.array(ml_views, np.float32),
            np.array(cameras, np.float32),
            None,
        )


class ResampledScene(AbstractScene):
    def __init__(self, original_scene, views):
        self.original_scene = original_scene
        self.views = views

    @property
    def name(self):
        return "resampled" + self.original_scene.name

    @property
    def images(self):
        return np.take(self.original_scene.images, self.views, axis=0)

    @property
    def gt_views(self):
        return np.take(self.original_scene.gt_views, self.views, axis=0)

    @property
    def ml_views(self):
        return np.take(self.original_scene.ml_views, self.views, axis=0)

    @property
    def cameras(self):
        return np.take(self.original_scene.cameras, self.views, axis=0)

    @property
    def rectangle(self):
        return self.original_scene.rectangle


def shortest_quad_side(quads: np.ndarray) -> Any:
    return np.amin(np.linalg.norm(np.roll(quads, 1, -2) - quads, axis=-1), -1)


# pylint: disable=no-member
def draw_quads(
    img: np.ndarray, quads: np.ndarray, color: tuple, thickness: int = 3
) -> None:
    pts = quads * [1, -1] + [0, IMAGE_SHAPE[1]]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(img, [pts], True, color, thickness)


# pylint: disable=no-member
def _debug_draw_detections(
    images: np.ndarray, aruco_views: np.ndarray, ml_views: np.ndarray
) -> None:
    for index, (img, aruco_view, ml_view) in enumerate(
        zip(images, aruco_views, ml_views)
    ):
        draw_quads(img, aruco_view, (255, 0, 0))
        draw_quads(img, ml_view, (0, 0, 255))
        cv2.imwrite(f"debug/{index}.png", img)


def resample_scenes(
    scenes: list[FullScene], samples: int, sample_state_size: int = None
) -> list[ResampledScene]:
    resampled = []
    for _ in range(samples):
        scene = random.choice(scenes)

        scene_views = list(range(scene.number_of_views))
        if sample_state_size is None:
            sampled_states = random.sample(
                scene_views, k=random.randint(2, min(15, scene.number_of_views))
            )
        else:
            sampled_states = random.sample(scene_views, k=sample_state_size)

        resampled.append(ResampledScene(scene, sampled_states))
    resampled.sort(key=lambda x: x.number_of_views)
    return resampled


def draw_error_sigma(
    methods: dict,
    scenes: list,
    output: Path,
):
    data = get_graph_data(methods,
                          scenes,
                          lambda s: s.diversity > 0.03 and s.sigma < 0.08,
                          output,
                          10,
    )

    for method_name, method_data in data.items():
        if method_name == 'None':
            method_name = 'Baseline'
        sigmas, errors = method_data["sigmas"], method_data["errors"]
        coeffs = np.linalg.lstsq(
            np.stack([sigmas, np.ones_like(sigmas)], axis=1),
            np.array(errors)[..., None],
        )[0]
        newx = np.linspace(0, 0.08, 100)
        newy = np.array(list(map(lambda x: coeffs[0] * x + coeffs[1], newx)))

        # err = [coeffs[0]*s+coeffs[1] - e for s, e in zip(sigmas, errors)]
        # plt.scatter(sigmas, err, s=2, marker='.', label=method_name)
        plt.scatter(sigmas, errors, s=1/5000, marker=',', alpha=.15)
        plt.plot(newx, newy, label=method_name)
    plt.xlabel("Keypoint detection error ($\\sigma$)")
    plt.ylabel("Screen reconstruction error ($E_{sr}$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.clf()

    scene_frequency = data["None"]["frequencies"]
    total = sum(scene_frequency.values())
    for scene_name, scene_freq in sorted(scene_frequency.items()):
        print(scene_name, scene_freq / total)


def draw_error_diversity(
    methods: dict,
    scenes: list,
    output: Path,
):
    data = get_graph_data(methods,
                          scenes,
                          lambda s: s.diversity > 1e-4 and s.sigma < 0.1,
                          output,
                          10,
    )

    for method_name, method_data in data.items():
        diversities, errors = np.array(method_data["diversities"]), np.array(
            method_data["errors"]
        )
        newx = np.linspace(0, 0.25)
        newx, newy, _ = loess_1d(diversities, errors, newx, frac=0.45)

        # err = [coeffs[0]*s+coeffs[1] - e for s, e in zip(sigmas, errors)]
        # plt.scatter(sigmas, err, s=2, marker='.', label=method_name)
        plt.scatter(diversities, errors, s=1/5000, marker=',', alpha=.15)
        plt.plot(newx, newy, label=method_name)
    plt.xlabel("View diversity ($R$)")
    plt.ylabel("Screen reconstruction error ($E_{sr}$)")
    plt.vlines([s.diversity for s in scenes], [0]*10, [35]*10, linestyles='--', )
    plt.ylim([0, 35])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.clf()

    scene_frequency = data["None"]["frequencies"]
    total = sum(scene_frequency.values())
    for scene_name, scene_freq in scene_frequency.items():
        print(scene_name, scene_freq / total)


def get_graph_data(
    methods: dict, scenes: list, filter_fn: Callable, output: Path, fixed_k=None
):
    output = output.with_suffix(".pickle")
    if output.exists():
        with output.open("rb") as fp:
            return pickle.load(fp)
    resampled = [s for s in resample_scenes(scenes, 20000, fixed_k) if filter_fn(s)]
    frequencies = {scene.name: 0 for scene in resampled}

    for scene in resampled:
        frequencies[scene.name] += 1

    data = {}
    for name, config in methods.items():
        errors = []
        implementation = implementation_from_config(config)
        for scene in tqdm(resampled):
            estimate = implementation(
                tf.constant(scene.ml_views), tf.constant(scene.cameras)
            )
            error = (
                100
                * _maximum_3d_error(
                    results=tf.constant(estimate)[None, ...],
                    expected=scene.rectangle[None, ...],
                ).numpy()
            )
            errors.append(error)
            frequencies[scene.name] += 1

        data[name] = {
            "errors": errors,
            "lengths": [s.number_of_views for s in resampled],
            "diversities": [s.diversity for s in resampled],
            "sigmas": [s.sigma for s in resampled],
            "frequencies": frequencies,
        }
    with output.open("wb") as fp:
        pickle.dump(data, fp)
    return data


def draw_error_length(
    methods: dict,
    scenes: list,
    output: Path,
):
    data = get_graph_data(
        methods, scenes, lambda s: s.sigma < 0.1 and s.diversity > 0.03, output
    )

    for method_name, method_data in data.items():
        if method_name == 'None':
            method_name = 'Baseline'
        lengths, errors = np.array(method_data["lengths"]), np.array(
            method_data["errors"]
        )
        new_lengths = []
        new_errors = []
        errors_stddev = []
        for length in sorted(set(lengths)):
            new_lengths.append(length)
            err = []
            for length_1, e in zip(lengths, errors):
                if length_1 == length:
                    err.append(e)
            new_errors.append(np.mean(err))
            errors_stddev.append(np.std(err))
        new_errors = np.array(new_errors)
        errors_stddev = np.array(errors_stddev)
        plt.plot(new_lengths, new_errors, label=method_name)
        #plt.fill_between(new_lengths, new_errors-errors_stddev, new_errors+errors_stddev, alpha=0.25)

    plt.xlabel("Number of views ($k$)")
    plt.ylabel("Screen reconstruction error ($E_{sr}$)")
    plt.xlim([2, 15])
    plt.ylim([0, 15])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)

    scene_frequency = data["None"]["frequencies"]
    total = sum(scene_frequency.values())
    for scene_name, scene_freq in scene_frequency.items():
        print(scene_name, scene_freq / total)


def characterize(scenes: list[FullScene]) -> None:
    sigmas = []
    reprojection_errors = []
    ks = []
    diversities = []
    assumption1 = []
    assumption2 = []
    for scene in scenes:
        sigmas.append(f"{scene.sigma:.2}")
        reprojection_error = np.round(
            np.sqrt(squared_reprojection_error(
                scene.rectangle,
                scene.gt_views,
                scene.cameras,
            )),
            1,
        )
        reprojection_errors.append(f"{reprojection_error:.2}")
        ks.append(scene.number_of_views)
        diversities.append(f"{scene.diversity:.2}")
        assumption1.append(scene.t_squared)
        assumption2.append(scene.likelihood_ratio)
    print("$k$", *ks, sep=" & ", end="\\\\\n")
    print("$\\sqrt{E_r}$", *reprojection_errors, sep=" & ", end="\\\\\n")
    print("$\\sigma$", *sigmas, sep=" & ", end="\\\\\n")
    print("$R$", *diversities, sep=" & ", end="\\\\\n")
    print("mean R, stddev", np.mean([s.diversity for s in scenes]), np.std([s.diversity for s in scenes]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-i",
        help="Path to the dataset of scenes to process.",
        type=Path,
        dest="dataset",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config_file", type=Path, required=True)
    parser.add_argument("--style_file", type=Path, required=True)
    args = parser.parse_args()

    scenes = []
    for scene_folder in sorted(args.dataset.iterdir(), key=lambda x: int(x.stem)):
        scenes.append(FullScene.from_path(scene_folder))

    config = _load_config(args.config_file)
    methods = _parse_methods(config["methods"])
    plt.style.use(['science', 'ieee', args.style_file])

    characterize(scenes)
    draw_error_sigma(methods, scenes, args.output / "rw_sigma.pdf")
    draw_error_diversity(methods, scenes, args.output / "rw_diversity.pdf")
    draw_error_length(methods, scenes, args.output / "rw_length.pdf")


if __name__ == "__main__":
    main()

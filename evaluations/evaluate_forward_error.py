from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

from stereo.evaluations.data import serve_data_with_error
from stereo.evaluations.utils import read_keypoints
from stereo.implementations import IMPLEMENTATIONS, LOSSES, Optimization, squared_reprojection_error
from stereo.implementations.utils.homogenous_coordinates import (
    from_homo_batched,
    to_homo_batched,
)


def count_within_threshold(
    results: np.ndarray,
    expected: np.ndarray,
    threshold: float,
) -> tf.Tensor:
    tf.debugging.assert_shapes(
        [
            (results, ("N", 4, 3)),
            (expected, ("N", 4, 3)),
        ]
    )
    height = tf.norm(expected[:, 3, :] - expected[:, 0, :], axis=-1)
    width = tf.norm(expected[:, 1, :] - expected[:, 0, :], axis=-1)
    shortest_side_length = tf.minimum(width, height)
    return tf.reduce_sum(
        tf.cast(
            tf.reduce_all(
                tf.math.less(
                    tf.norm(results - expected, axis=-1)
                    / shortest_side_length[:, tf.newaxis],
                    threshold,
                ),
                axis=-1,
            ),
            tf.uint16,
        )
    )


def _maximum_3d_error(
    results: np.ndarray,  # (N, 4, 3)
    expected: np.ndarray,  # (N, 4, 3)
) -> tf.Tensor:
    height = tf.norm(expected[:, 3, :] - expected[:, 0, :], axis=-1)
    width = tf.norm(expected[:, 1, :] - expected[:, 0, :], axis=-1)
    shortest_side_length = tf.minimum(width, height)
    return tf.reduce_sum(
        tf.reduce_max(
            tf.norm(results - expected, axis=-1) / shortest_side_length[:, tf.newaxis],
            axis=-1,
        )
    )


def _mean_reprojection_error(
    results: np.ndarray,  # (B, N, 3)
    keypoint_state: np.ndarray,  # (B, T, N, 2)
    camera_state: np.ndarray,  # (B, T, 3, 4)
) -> tf.Tensor:
    X_homo = to_homo_batched(tf.constant(results))  # (B, N, 4)

    A = tf.einsum(
        "acx,abdx->abcd",
        X_homo,
        camera_state,
    )  # (B, T, N, 3)

    A = from_homo_batched(A)  # (B, T, N, 2)

    A = A - keypoint_state  # (B, T, N, 2)  # (B, T, N, 2)

    A = tf.norm(A, axis=-1)  # (B, T, N)
    return tf.math.sqrt(tf.nn.l2_loss(A)) / tf.cast(
        tf.shape(keypoint_state)[1], "float32"
    )


def evaluate(
    method: Optimization,
    samples: list[tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
    trials: int,
) -> tuple[float, float]:
    rectangles: np.ndarray = np.array([sample[0] for sample in samples])
    keypoint_state: np.ndarray = np.array([sample[1] for sample in samples])
    camera_state: np.ndarray = np.array([sample[2] for sample in samples])

    results: np.ndarray = np.array([method(*sample[1:]) for sample in samples])
    result = _maximum_3d_error(results=results, expected=rectangles)
    mre = tf.reduce_mean(tf.vectorized_map(lambda x: squared_reprojection_error(*x), (results, keypoint_state, camera_state)))
    return result / trials, mre


def run_trial(
    methods: dict[str, dict[str, Any]],
    rectangles: tf.Tensor,
    keypoint_states: tf.Tensor,
    camera_states: tf.Tensor,
    sigmas: np.ndarray,
    trials: int = 10,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    # pylint: disable=too-many-locals
    results: dict[str, list[float]] = {}
    mres: dict[str, list[float]] = {}

    for name, config in methods.items():
        implementation = IMPLEMENTATIONS[config.pop("implementation")]
        state_length = config.pop("state_length")
        results[name] = []
        mres[name] = []
        if config:
            implementation.set_hyperparameters(**config)
        print(f"Evaluating method {name}")
        for sigma in tqdm(sigmas):
            score, mre = evaluate(
                implementation,
                list(
                    serve_data_with_error(
                        sigma=sigma,
                        trials=trials,
                        sample_state_length=state_length,
                        full_rectangles=rectangles,
                        full_keypoint_states=keypoint_states,
                        full_camera_states=camera_states,
                    )
                ),
                trials,
            )
            results[name].append(float(score))
            mres[name].append(float(mre))
    return results, mres


def draw_plot(
    sigmas: np.ndarray,
    result: dict[str, list[float]],
    filename: Path,
    y_label: str,
    y_percentage: bool = False,
) -> None:
    for method, y_values in result.items():
        x_values = sigmas * 100
        if y_percentage:
            y_values = [y * 100 for y in y_values]

        plt.plot(x_values, y_values, label=method)
    plt.xlabel("Error in 2D detections (%)")

    if y_percentage:
        plt.ylim([-5, 105])

    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(filename)
    plt.clf()
    plt.cla()


def _parse_method(method_config: dict) -> dict[str, Any]:
    if not method_config or "optimizer" not in method_config:
        return method_config

    optimizer_config = method_config.pop("optimizer")
    loss_function = LOSSES[method_config.pop("loss")]
    return {
        **method_config,
        "optimizer": tf.keras.optimizers.deserialize(optimizer_config),
        "loss_function": loss_function,
    }


def _parse_methods(methods_config: dict[str, dict]) -> dict[str, Any]:
    return {
        method_name: _parse_method(method_config)
        for method_name, method_config in methods_config.items()
    }


def _load_config(config_path: Path) -> Any:
    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def draw_plots(
    sigmas: np.ndarray, results: dict, mres: dict, output_path: Path
) -> None:
    fig = plt.figure()

    err_2d = fig.add_subplot(2, 1, 1)
    for method, y_values in mres.items():
        if method == 'None':
            method = 'Baseline'
        err_2d.plot(sigmas, y_values, label=method)
    err_2d.set_ylabel("Square root of reprojection error ($\\sqrt{E_r}$)")

    err_3d = fig.add_subplot(2, 1, 2)
    for method, y_values in results.items():
        if method == 'None':
            method = 'Baseline'
        err_3d.plot(sigmas, 100 * np.array(y_values), label=method)
    err_3d.set_ylabel("Screen reconstruction error ($E_{sr}$)")

    plt.xlabel("Keypoint detection error ($\\sigma$)")
    plt.legend()
    fig.tight_layout()
    plt.savefig(output_path)


def draw_3d_only(sigmas: np.ndarray, results: dict, output_path: Path) -> None:
    fig = plt.figure()

    err_3d = fig.add_subplot(1, 1, 1)
    for method, y_values in results.items():
        if method == 'None':
            method = 'Baseline'
        err_3d.plot(sigmas, 100 * np.array(y_values), label=method)
    err_3d.set_ylabel("Screen reconstruction error ($E_{sr}$)")

    plt.legend()
    plt.xlabel("Keypoint detection error ($\\sigma$)")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.cla()
    plt.clf()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style_file", type=Path, required=True)
    parser.add_argument("--config_file", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = _load_config(args.config_file)
    rectangles, camera_states, keypoint_states = read_keypoints(str(args.dataset))
    sigmas = np.linspace(
        config["x_axis_range"][0],
        config["x_axis_range"][1],
        config["x_axis_point_cnt"],
    )
    output_name = config["output_name"]
    results, mres = run_trial(
        methods=_parse_methods(config["methods"]),
        rectangles=rectangles,
        keypoint_states=keypoint_states,
        camera_states=camera_states,
        sigmas=sigmas,
        trials=config["trials"],
    )

    graph_path = args.output
    graph_path.mkdir(exist_ok=True, parents=True)

    plt.style.use(['science', 'ieee', args.style_file])

    draw_plots(sigmas, results, mres, graph_path / f"{output_name}_both.pdf")
    draw_3d_only(sigmas, results, graph_path / f"{output_name}_3D.pdf")


if __name__ == "__main__":
    main()

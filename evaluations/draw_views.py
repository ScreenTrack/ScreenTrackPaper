import argparse
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from vpython import arrow, canvas, quad, vector, vertex

from stereo.evaluations.utils import read_keypoints
from stereo.implementations.custom_optimization_procedure import get_rays_batched


def plot_keypoints(keypoint_state: np.ndarray) -> None:
    count = len(keypoint_state)
    fig = plt.figure()
    axe = fig.add_subplot(111)
    patches = [Polygon(kps, True) for kps in keypoint_state]
    collection = PatchCollection(patches)

    def opacity(color: tuple[float], alpha: float) -> tuple[float]:
        return color[:-1] + (alpha,)

    collection.set_facecolor(
        [opacity(matplotlib.cm.jet(i / count), 0.5) for i in range(count)]
    )
    axe.add_collection(collection)
    plt.xlim([0, 1920])
    plt.ylim([0, 1080])
    plt.xticks([])
    plt.yticks([])
    for patch in patches:
        axe.add_patch(Polygon(patch.get_xy(), closed=True, ec="k", lw=1, fill=False))
    plt.savefig("graphs/2d_views.png")


def plot_cameras(rectangle: np.ndarray, camera_state: np.ndarray) -> None:
    count = len(camera_state)
    camera_centers, rays = get_rays_batched(
        tf.tile([[[1920 / 2, 1080 / 2]]], [count, 4, 1]), tf.constant(camera_state)
    )
    _, up_rays = get_rays_batched(
        tf.tile([[[1920 / 2, 1e10]]], [count, 4, 1]), tf.constant(camera_state)
    )
    camera_centers = tf.squeeze(camera_centers).numpy()
    rays = rays[:, 0, :].numpy()
    up_rays = up_rays[:, 0, :].numpy()
    up_rays -= rays * np.sum(rays * up_rays, axis=-1)[:, None]
    up_rays /= np.linalg.norm(up_rays, axis=-1)[:, None]

    cvs = canvas(title="Cameras in 3D", width=1920, height=1080, background=vector(1, 1, 1))
    texpos = [vector(0, 0, 0), vector(1, 0, 0), vector(1, 1, 0), vector(0, 1, 0)]
    cvs.camera.axis = vector(*rays[0])
    cvs.camera.pos = vector(*camera_centers[0])
    cvs.camera.up = vector(*up_rays[0])
    quad(vs=[vertex(pos=vector(*p), texpos=tp)
             for p, tp in zip(rectangle, texpos)], texture={'file': 'gold.png'})

    for idx, (center, ray) in enumerate(zip(camera_centers, rays)):
        cvs.camera.axis = vector(*rays[idx])
        cvs.camera.pos = vector(*camera_centers[idx])
        cvs.camera.up = vector(*up_rays[idx])
        arrow(
            pos=vector(*center),
            axis=vector(*ray) / 3,
            color=vector(*[[0, 0, 0], [0, 0, 0]][idx%2]),
        )
    time.sleep(10)
    cvs.capture('all')


def main(
    input_file: Path,
    rectangle_id: int,
    draw_all: bool,
) -> None:
    rectangles, camera_states, keypoint_states = read_keypoints(str(input_file))
    sample = [6, 12]
    if draw_all:
        sample = np.arange(len(camera_states[rectangle_id]))
    keypoint_sample = np.take(keypoint_states[rectangle_id], sample, 0)
    camera_sample = np.take(camera_states[rectangle_id], sample, 0)
    plot_keypoints(keypoint_sample)
    plot_cameras(rectangles[rectangle_id], camera_sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw views of a single rectangle")
    parser.add_argument(
        "--input", type=Path, help="Database of rectangles to use", required=True
    )
    parser.add_argument(
        "--rectangle",
        type=int,
        default=0,
    )
    parser.add_argument(
        '--all', action='store_true'
    )
    args = parser.parse_args()
    main(args.input, args.rectangle, args.all)

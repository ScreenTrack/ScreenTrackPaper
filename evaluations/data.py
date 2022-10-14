from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

import numpy as np
import tensorflow as tf


def minimal_quadrilateral_side(quad: np.ndarray) -> Any:
    rolled_quad: np.ndarray = np.roll(quad, 1, axis=0)
    return np.min(np.linalg.norm(rolled_quad - quad, axis=-1))


def introduce_error_single(keypoints: np.ndarray, sigma: float) -> np.ndarray:
    sigma *= minimal_quadrilateral_side(keypoints)
    keypoints_with_error: np.ndarray = (
        keypoints
        + sigma
        * np.random.multivariate_normal(
            mean=(0, 0),
            cov=[[1, 0], [0, 1]],
            size=4,
        )
    )

    return keypoints_with_error


def introduce_error(sample_keypoint_state: np.ndarray, sigma: float) -> tf.Tensor:
    return tf.convert_to_tensor(
        [
            introduce_error_single(sample_keypoints, sigma=sigma)
            for sample_keypoints in sample_keypoint_state
        ]
    )


def serve_data_with_error(
    sigma: float,
    trials: int,
    full_rectangles: tf.Tensor,
    full_keypoint_states: tf.Tensor,
    full_camera_states: tf.Tensor,
    sample_state_length: int,
) -> Iterable[tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    tf.debugging.assert_shapes(
        [
            (full_rectangles, ("R", 4, 3)),
            (full_keypoint_states, ("R", "N", 4, 2)),
            (full_camera_states, ("R", "N", 3, 4)),
        ]
    )
    rectangle_count = full_rectangles.shape[0]
    full_state_length = full_keypoint_states.shape[1]

    data_indices = list(range(full_state_length))
    rectangle_indices = list(range(rectangle_count))

    for sampled_rectangle_index in random.sample(rectangle_indices, k=trials):
        rectangle = full_rectangles[sampled_rectangle_index]
        full_keypoint_state = full_keypoint_states[sampled_rectangle_index]
        full_camera_state = full_camera_states[sampled_rectangle_index]

        sampled_indices = random.sample(
            data_indices,
            k=sample_state_length,
        )
        # pylint: disable=no-value-for-parameter
        yield (
            rectangle,
            introduce_error(
                tf.gather(full_keypoint_state, sampled_indices),
                sigma,
            ),
            tf.gather(full_camera_state, sampled_indices),
        )

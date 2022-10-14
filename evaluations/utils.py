from __future__ import annotations

import numpy as np
import tensorflow as tf

from stereo.implementations import IMPLEMENTATIONS, Optimization


def read_keypoints(filename: str) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    with open(filename, "rb") as file:
        rectangles = tf.constant(np.load(file), dtype=tf.float32)
        camera_states = tf.constant(np.load(file), dtype=tf.float32)
        keypoint_states = tf.constant(np.load(file), dtype=tf.float32)
    return rectangles, camera_states, keypoint_states


def implementation_from_config(config: dict) -> Optimization:
    config = config.copy()
    implementation = IMPLEMENTATIONS[config.pop("implementation")]
    if "state_length" in config:
        config.pop("state_length")
    if config:
        implementation.set_hyperparameters(**config)
    return implementation

from __future__ import annotations

import tensorflow as tf


# fmt: off
# (B, N, 3) -> (B, N, 4)
@tf.function
def to_homo_batched(X: tf.Tensor) -> tf.Tensor:
    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    target_shape = tf.concat((tf.shape(X)[:-1], tf.constant([1])), axis=-1)
    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    return tf.concat(
        (
            X,
            tf.ones(target_shape, dtype=tf.float32)
        ),
        axis=-1,
    )


# (3,) -> (4,)
@tf.function
def to_homo(X: tf.Tensor) -> tf.Tensor:
    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    return tf.concat(
        (X, tf.constant([1], dtype=tf.float32)),
        axis=-1,
    )


# (T, 3) -> (T, 2)
@tf.function
def from_homo(X: tf.Tensor) -> tf.Tensor:
    inhomo = X[:, :-1]  # (3, 2)
    scale = X[:, -1]  # ()
    return inhomo / scale[:, None]


# (B, N, 4) -> (B, N, 3)
@tf.function(experimental_relax_shapes=True)
def from_homo_batched(X: tf.Tensor) -> tf.Tensor:
    inhomo = X[..., :-1]  # (N, 3)
    scale = X[..., -1, None]  # (N, 1)
    return inhomo / scale

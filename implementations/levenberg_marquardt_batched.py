from __future__ import annotations

import tensorflow as tf
from tensorflow_graphics.math.optimizer import (
    levenberg_marquardt as levenberg_marquardt_optimizer,
)

from stereo.implementations.utils.homogenous_coordinates import (
    from_homo_batched as from_homo,
)
from stereo.implementations.utils.homogenous_coordinates import (
    to_homo_batched as to_homo,
)


@tf.function
def reconstruct_3d_inhomogenous(
    keypoint_state: tf.Tensor,
    camera_state: tf.Tensor,
) -> tf.Tensor:
    """
    Reconstructs 3D points using the inhomogeneous method.
    This method is based on the fact that the projection matrix P
    operating on a 3D point in homogeneous coordinates outputs the
    2D image point in homogeneous coordinates.

    Let the 2D image point be (u, v), X point in 3D space,
    and P the projection matrix.

    We have PX=x where P is a 3x4 matrix,
    X=(x,y,z,w), x=(tu, tv, t) where t is unknown.

    Since the point X is not at the plane at infinity we take w=1.
    To remove the unknown t we use the cross product to get:
        PX x (u, v, 1) = 0

    This gives us a system of three equations with three unknowns -
    however the rank of the system is 2, so we need at least two
    images to detect points in 3D space.

    With 2 or more images the system of equations is overdetermined
    so we use the linear least squares method.
    :param keypoint_state: (T, N, 2)
    :param camera_state: (T, 3, 4)
    :return: (N, 3)
    """
    # fmt: off
    A = (
            keypoint_state[..., None]  # (B, T, N, 2, 1)
            * camera_state[:, :, None, None, 2, :]  # (B, T, 1, 1, 4)
    )  # (B, T, N, 2, 4)

    A = (
            A -
            camera_state[:, :, None, :2, :]  # (B, T, 1, 2, 4)
    )  # (B, T, N, 2, 4)

    tf.debugging.assert_shapes(
        [
            (keypoint_state, ("B", "T", "N", 2)),
            (camera_state, ("B", "T", 3, 4)),
            (A, ("B", "T", "N", 2, 4)),
        ]
    )

    A = tf.transpose(A, (0, 2, 1, 3, 4))  # (B, N, T, 2, 4)
    shape = tf.shape(A)
    A = tf.reshape(A, (shape[0], shape[1], -1, 4))  # (B, N, 2T, 4)

    B = -A[..., -1, None]  # (B, N, 2T, 1)
    A = A[..., :-1]  # (B, N, 2T, 3)

    # pylint: disable=no-value-for-parameter
    X = tf.stop_gradient(tf.linalg.lstsq(A, B))  # (B, N, 3, 1)
    tf.debugging.assert_shapes(
        [
            (keypoint_state, ("B", "T", "N", 2)),
            (X, ("B", "N", 3, 1)),
        ]
    )
    X = tf.squeeze(X, (-1))
    return X  # (B, N, 3)
    # fmt: on


@tf.function
def levenberg_marquardt(
    X0: tf.Tensor,
    keypoint_state: tf.Tensor,
    camera_state: tf.Tensor,
) -> tf.Tensor:
    """
    This method minimises the reprojection error of the
    point in 3D space. Reprojection error of the point X
    in the image given by the projective matrix P is the
    square of the distance of the point PX to the actual
    detected point in the image.

    The Levenberg-Marquardt method allows minimisation of
    squared errors of arbitrary functions so we can use
    the reprojection errors in the N collected images as
    the value to be minimised.

    Most of the complexity in this method comes from the
    fact that the projection matrix operates on points in
    homogeneous coordinates on which the L_2 norm doesn't
    make sense.
    Therefore the points are converted from and to
    homogeneous coordinates,

    :param X0: (N, 3)
    :param keypoint_state: (T, N, 2)
    :param camera_state: (T, 3, 4)
    :return: (N, 3)
    """

    tf.debugging.assert_shapes(
        [
            (keypoint_state, ("B", "T", "N", 2)),
            (camera_state, ("B", "T", 3, 4)),
            (X0, ("B", "N", 3)),
        ]
    )

    def fun(X: tf.Tensor) -> tf.Tensor:  # (B, N, 3) -> (B, N, 1)
        X_homo = to_homo(X)  # (B, N, 4)

        A = tf.squeeze(
            (
                camera_state[:, None, ...]  # (B, 1, T, 3, 4)
                @ X_homo[:, :, None, ..., None]  # (B, N, 1, 4, 1)
            ),  # (B, N, T, 3, 1)
            (-1),
        )  # (B, N, T, 3)

        A = from_homo(A)  # (B, N, T, 2)

        A = A - tf.transpose(  # (B, N, T, 2)
            keypoint_state, (0, 2, 1, 3)
        )  # (B, N, T, 2)  # (B, N, T, 2)

        A_shape = tf.shape(A)
        return tf.reshape(A, (A_shape[0] * A_shape[1], -1))  # (B, N, 2T)

    _, (final_variables,) = levenberg_marquardt_optimizer.minimize(
        residuals=fun,
        variables=X0,
        max_iterations=10,
    )
    # fmt: on

    tf.debugging.assert_shapes(
        [
            (keypoint_state, ("B", "T", "N", 2)),
            (final_variables, ("B", "N", 3)),
        ]
    )
    return tf.stop_gradient(final_variables)

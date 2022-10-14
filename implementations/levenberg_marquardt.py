from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf
from tensorflow_graphics.math.optimizer import (
    levenberg_marquardt as levenberg_marquardt_optimizer,
)

from stereo.implementations.utils.homogenous_coordinates import (
    from_homo_batched,
    to_homo_batched,
)


@tf.function(
    input_signature=[
        (
            tf.TensorSpec((None, 4, 2), tf.float32),
            tf.TensorSpec((None, 3, 4), tf.float32),
        )
    ]
)
def reconstruct_3d_inhomogenous(
    inputs: tuple[tf.Tensor, tf.Tensor],
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
    """
    keypoint_state, camera_state = inputs

    tf.debugging.assert_shapes(
        [
            (keypoint_state, ("T", "N", 2)),
            (camera_state, ("T", 3, 4)),
        ]
    )

    A = (
        keypoint_state[..., None]  # (T, N, 2, 1)
        * camera_state[:, None, None, 2, :]  # (T, 1, 1, 4)
    )  # (T, N, 2, 4)

    A = A - camera_state[..., None, :2, :]  # (T, 1, 2, 4)  # (T, N, 2, 4)

    tf.debugging.assert_shapes(
        [
            (keypoint_state, ("T", "N", 2)),
            (camera_state, ("T", 3, 4)),
            (A, ("T", "N", 2, 4)),
        ]
    )

    A = tf.transpose(A, (1, 0, 2, 3))  # (N, T, 2, 4)
    shape = tf.shape(A)
    A = tf.reshape(A, (shape[0], -1, 4))  # (N, 2T, 4)

    B = -A[..., -1, None]  # (N, 2T, 1)
    A = A[..., :-1]  # (N, 2T, 3)

    # pylint: disable=no-value-for-parameter
    X = tf.stop_gradient(tf.linalg.lstsq(A, B))  # (N, 3, 1)
    tf.debugging.assert_shapes(
        [
            (keypoint_state, ("T", "N", 2)),
            (X, ("N", 3, 1)),
        ]
    )
    X = tf.squeeze(X, (-1))
    return X  # (N, 3)


@tf.function
def reproject_batched(
    positions_in_3d: tf.Tensor,  # (N, 3)
    camera_state: tf.Tensor,  # (T, 3, 4)
) -> tf.Tensor:
    homo_positions_in_3d = to_homo_batched(positions_in_3d)  # (N, 4)
    homo_projected = tf.transpose(
        tf.tensordot(
            homo_positions_in_3d,  # (N, 4)
            camera_state,  # (T, 3, 4)
            axes=[[1], [2]],
        ),
        [1, 0, 2],
    )
    projected = from_homo_batched(homo_projected)
    return projected


@tf.function
def levenberg_marquardt(
    inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
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
    """
    X0, keypoint_state, camera_state = inputs
    tf.debugging.assert_shapes(
        [
            (X0, ("N", 3)),
            (keypoint_state, ("T", "N", 2)),
            (camera_state, ("T", 3, 4)),
        ]
    )

    # fmt: off
    def create_reprojection_diff(
        kpt_id: tf.Tensor,
    ) -> Callable[[tf.Tensor], tf.Tensor]:
        def reprojection_diff(
            point_in_3d: tf.Tensor,
        ) -> tf.Tensor:  # (3,) -> (2T,)
            reprojected = tf.squeeze(
                reproject_batched(point_in_3d[None, ...], camera_state),
                1,
            )
            error = reprojected - keypoint_state[:, kpt_id, :]
            sample_count = tf.shape(keypoint_state)[0]
            return tf.math.sqrt(tf.nn.l2_loss(error)) / tf.cast(sample_count, "float32")

        return reprojection_diff

    num_keypoints = tf.shape(keypoint_state)[-2]

    final_variables = tf.zeros((num_keypoints, 3), dtype=tf.float32)
    for keypoint_id in tf.range(num_keypoints):
        _, (final_variables_for_kpt,) = levenberg_marquardt_optimizer.minimize(
            residuals=create_reprojection_diff(keypoint_id),
            variables=X0[keypoint_id],
            max_iterations=10,
        )

        final_variables = tf.tensor_scatter_nd_update(
            final_variables,
            tf.transpose(
                tf.stack(
                    [
                        tf.repeat(keypoint_id, repeats=3, axis=0),
                        tf.range(3),
                    ],
                    axis=0,
                )
            ),
            final_variables_for_kpt,
        )

    # fmt: on

    tf.debugging.assert_shapes(
        [
            (keypoint_state, ("T", "N", 2)),
            (final_variables, ("N", 3)),
        ]
    )

    return tf.stop_gradient(final_variables)


@tf.function(experimental_relax_shapes=True)
def parametrized_levenberg_marquardt(
    steps: int,
    keypoint_state: tf.Tensor,
    camera_state: tf.Tensor,
    initial_estimate_params: tf.Tensor,
    parametrization_to_keypoints: Callable,
    regularizer: float = 1e-20,
    regularizer_multiplier: float = 10.0,
) -> tf.Tensor:
    def residual(keypoint_params: tf.Tensor) -> tf.Tensor:
        error = (
            reproject_batched(
                parametrization_to_keypoints(keypoint_params), camera_state
            )
            - keypoint_state
        )
        sample_count = tf.shape(keypoint_state)[0]
        return tf.math.sqrt(tf.nn.l2_loss(error)) / tf.cast(sample_count, "float32")

    _, (estimate_params,) = levenberg_marquardt_optimizer.minimize(
        residuals=residual,
        variables=initial_estimate_params,
        max_iterations=steps,
        regularizer=regularizer,
        regularizer_multiplier=regularizer_multiplier,
    )
    return tf.stop_gradient(parametrization_to_keypoints(estimate_params))

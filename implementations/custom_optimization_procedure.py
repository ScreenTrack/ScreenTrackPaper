from __future__ import annotations

from collections.abc import Callable

import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import axis_angle, quaternion

from stereo.implementations.utils.homogenous_coordinates import (
    from_homo_batched as from_homo,
)
from stereo.implementations.utils.homogenous_coordinates import to_homo_batched


@tf.function
def _dimensions_center_and_rotation_matrix(position_3d: tf.Tensor) -> tf.Tensor:
    width_vector = (
        (position_3d[1] - position_3d[0]) + (position_3d[2] - position_3d[3])
    ) / 2
    height_vector = (
        (position_3d[3] - position_3d[0]) + (position_3d[2] - position_3d[1])
    ) / 2
    center = position_3d[0] + width_vector / 2 + height_vector / 2

    rotx, width = tf.linalg.normalize(width_vector)
    roty, height = tf.linalg.normalize(height_vector)
    rotz = tf.linalg.cross(rotx, roty)

    rotation_matrix = tf.stack([rotx, roty, rotz], axis=1)
    return width, height, center, rotation_matrix


@tf.function
def _get_scaled_unit_rectangle(width: tf.Tensor, height: tf.Tensor) -> tf.Tensor:
    return tf.constant(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]
    ) * [width / 2, height / 2, 1]


@tf.function
def decode_8_params(
    keypoint_params: tf.Variable,
) -> tf.Tensor:
    """
    keypoint_params = (*left_bottom, *right_bottom, *left_top_yz)  # (8,)

    normal = (a, b, c)
    left_bottom = (x_0, y_0, z_0)

    plane ... a(x-x_0) + b(y-y_0) + c(z-z_0) = 0
              x = x0 - (b(y-y_0) - c(z-z0))/a
    """
    left_bottom = keypoint_params[:3]
    right_bottom = keypoint_params[3:6]
    left_top_yz = keypoint_params[-2:]

    normal = right_bottom - left_bottom

    left_top_x = (
        left_bottom[0]
        - tf.reduce_sum(normal[1:] * (left_top_yz - left_bottom[1:])) / normal[0]
    )

    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    left_top = tf.concat([left_top_x[None, ...], left_top_yz], axis=0)

    right_top = right_bottom + (left_top - left_bottom)

    return tf.stack(
        [left_bottom, right_bottom, right_top, left_top],
        axis=0,
    )


@tf.function
def encode_8_params(keypoints: tf.Tensor) -> tf.Tensor:
    return tf.concat([keypoints[0], keypoints[1], keypoints[3][1:]], 0)


@tf.function
def decode_9_params(
    keypoint_params: tf.Variable,
) -> tf.Tensor:
    """
    keypoint_params = (*left_bottom, *right_bottom, *left_top)  # (9,)
    """
    left_bottom = keypoint_params[:3]
    right_bottom = keypoint_params[3:6]
    left_top = keypoint_params[-3:]
    right_top = right_bottom + (left_top - left_bottom)

    return tf.stack(
        [left_bottom, right_bottom, right_top, left_top],
        axis=0,
    )


@tf.function
def encode_9_params(position_3d: tf.Tensor) -> tf.Tensor:
    return tf.concat([position_3d[0], position_3d[1], position_3d[3]], 0)


# empirical constant todo recalculate w optuna
RIGID_BODY_QUAT_ROTATION_SCALE = 1


@tf.function
def decode_rigid_body_quat(rigid_body_params: tf.Variable) -> tf.Tensor:
    """
    rigid_body_params = (width, height, qi, qj, qk, q, x, y, z)
    The rectangle is obtained by scaling the rectangle
    (-1, -1, 0)------(1, 1, 0)
         |               |
         |               |
         |               |
    (-1, -1, 0)------(1, -1, 0)
    by width in the direction of x axis, and by height in the direction of y axis. Then
    the points are rotated by the quaternion (qi, qj, qk, q), and finally translated by
    (x, y, z)
    """
    # Scale
    width = rigid_body_params[0]
    height = rigid_body_params[1]
    keypoint_params = _get_scaled_unit_rectangle(width, height)

    # Rotate
    rotation = quaternion.normalize(
        rigid_body_params[2:6] / RIGID_BODY_QUAT_ROTATION_SCALE
    )
    keypoint_params = quaternion.rotate(keypoint_params, rotation)

    # Translate
    keypoint_params += rigid_body_params[6:9]

    return keypoint_params


@tf.function
def encode_rigid_body_quat(position_3d: tf.Tensor) -> tf.Tensor:
    width, height, center, rotation_matrix = _dimensions_center_and_rotation_matrix(
        position_3d
    )
    rotation_quaternion = quaternion.from_rotation_matrix(rotation_matrix)

    return tf.concat(
        [width, height, rotation_quaternion * RIGID_BODY_QUAT_ROTATION_SCALE, center], 0
    )


# empirical constant
RIGID_BODY_ROTATION_SCALE = 1


@tf.function
def decode_rigid_body(rigid_body_params: tf.Variable) -> tf.Tensor:
    """
    rigid_body_params = (width, height, rx, ry, rz, x, y, z)
    The rectangle is obtained by scaling the rectangle
    (-1, -1, 0)------(1, 1, 0)
         |               |
         |               |
         |               |
    (-1, -1, 0)------(1, -1, 0)
    by width in the direction of x axis, and by height in the direction of y axis. Then
    the points are rotated around the axis (rx, ry, rz) by the angle equal to
    the norm of the axis vector, and finally points are translated by (x, y, z).
    """
    # Scale
    width = rigid_body_params[0]
    height = rigid_body_params[1]
    keypoint_params = _get_scaled_unit_rectangle(width, height)

    # Rotate
    descaled_rotation = rigid_body_params[2:5] / RIGID_BODY_ROTATION_SCALE
    axis, theta = tf.linalg.normalize(descaled_rotation)
    keypoint_params = axis_angle.rotate(keypoint_params, [axis], [theta])

    # Translate
    keypoint_params += rigid_body_params[5:8]

    return keypoint_params


@tf.function
def encode_rigid_body(position_3d: tf.Tensor) -> tf.Tensor:
    width, height, center, rotation_matrix = _dimensions_center_and_rotation_matrix(
        position_3d
    )
    axis, angle = axis_angle.from_rotation_matrix(rotation_matrix)

    return tf.concat(
        [width, height, axis * angle * RIGID_BODY_ROTATION_SCALE, center],
        0,
    )


@tf.function
def reproject(
    estimated_keypoints: tf.Tensor,  # (N, 3)
    camera_state: tf.Tensor,  # (T, 3, 4)
) -> tf.Tensor:  # (N, 3) -> (N, 2T)
    estimated_keypoints_homo = to_homo_batched(estimated_keypoints)  # (N, 4)
    projected_keypoints = tf.squeeze(
        (
            camera_state[:, None, ...]  # (T, 1, 3, 4)
            @ estimated_keypoints_homo[None, ..., None]  # (1, N, 4, 1)
        ),  # (T, N, 3, 1)
        (-1),
    )  # (T, N, 3)
    return from_homo(projected_keypoints)  # (T, N, 2)


@tf.function(
    input_signature=(
        tf.TensorSpec((4, 3), tf.float32),
        tf.TensorSpec((None, 4, 2), tf.float32),
        tf.TensorSpec((None, 3, 4), tf.float32),
    )
)
def squared_reprojection_error(
    estimated_keypoints: tf.Tensor,  # (4, 3)
    keypoint_state: tf.Tensor,  # (T, 4, 2)
    camera_state: tf.Tensor,  # (T, 3, 4)
) -> tf.Tensor:
    estimated = reproject(estimated_keypoints, camera_state)
    ground_truth = keypoint_state
    norm = tf.norm(estimated - ground_truth, axis=-1)

    state_length = tf.shape(estimated)[1]
    return tf.math.sqrt(tf.reduce_sum(norm**2)) / tf.cast(state_length, "float32")


@tf.function
def get_rays_batched(
    keypoint_state: tf.Tensor,  # (T, 4, 3)
    camera_state: tf.Tensor,  # (T, 3, 4)
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Let p be a camera (3, 4)
    p = [m, p4] where m is (3, 3), and p4 is (3, 1)

    m @ c = - p4 where c is (3,) center of the camera
    (Multiple view geometry p. 162)

    the line of the point x = (u, v, 1)^T is given by
    X(mu) = mu (m ^ -1 x, 0)^T + (c, 1)^T
    """
    # pylint: disable=invalid-name
    tf.debugging.assert_shapes(
        [
            (keypoint_state, ("T", 4, 2)),
            (camera_state, ("T", 3, 4)),
        ]
    )
    homo_2d_points = to_homo_batched(keypoint_state)  # (T, 4, 3)

    homo_2d_points_t = tf.transpose(homo_2d_points, perm=[0, 2, 1])
    # (T, 3, 4)

    # Solve the linear system m @ x = b
    m, p4 = tf.split(camera_state, [3, 1], -1)  # (T, 3, 3), (T, 3, 1)
    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    b = tf.concat([-p4, homo_2d_points_t], axis=-1)  # (T, 3, 5)
    x = tf.linalg.solve(m, b)  # (T, 3, 5)

    # pylint: disable=no-value-for-parameter
    camera_centers, ray_directions = tf.split(
        tf.transpose(x, perm=[0, 2, 1]), [1, 4], -2
    )  # (T, 1, 3), (T, 4, 3)

    return camera_centers, ray_directions


@tf.function
def angular_reprojection_error(
    estimated_keypoints: tf.Tensor,  # (4, 3)
    keypoint_state: tf.Tensor,  # (T, 4, 2)
    camera_state: tf.Tensor,  # (T, 3, 4)
) -> tf.Tensor:
    tf.debugging.assert_shapes(
        [
            (estimated_keypoints, (4, 3)),
            (keypoint_state, ("T", 4, 2)),
            (camera_state, ("T", 3, 4)),
        ]
    )

    camera_center, reprojected_ray = get_rays_batched(keypoint_state, camera_state)

    actual_ray = estimated_keypoints[tf.newaxis, ...] - tf.tile(
        camera_center, [1, 4, 1]
    )  # (T, 4, 3)

    actual_ray, _ = tf.linalg.normalize(actual_ray, axis=-1)  # (T, 4, 1)
    reprojected_ray, _ = tf.linalg.normalize(reprojected_ray, axis=-1)  # (T, 4, 1)
    # values for acos can't be clipped to [-1, 1] because the gradient
    # is nan for acos(1)
    theta = tf.math.acos(
        tf.clip_by_value(
            tf.reduce_sum(actual_ray * reprojected_ray, axis=-1), -1 + 2e-8, 1 - 2e-8
        )
    )  # (T, 4)

    sample_count = tf.shape(keypoint_state)[0]
    point_count = tf.cast(4 * sample_count, "float32")
    return tf.math.reduce_sum(tf.reshape(theta, [-1])) / point_count


def train(
    steps: int,
    keypoint_state: tf.Tensor,
    camera_state: tf.Tensor,
    initial_estimate_params: tf.Tensor,
    parametrization_to_keypoints: Callable,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_function: Callable,
) -> tf.Tensor:
    keypoint_params = tf.Variable(
        initial_value=initial_estimate_params, dtype=tf.float32
    )
    best_keypoint_params = tf.Variable(initial_value=keypoint_params, dtype=tf.float32)
    best_loss = tf.Variable(
        initial_value=loss_function(
            estimated_keypoints=parametrization_to_keypoints(keypoint_params),
            keypoint_state=keypoint_state,
            camera_state=camera_state,
        ),
        dtype=tf.float32,
    )
    return _train(
        steps,
        keypoint_params,
        best_keypoint_params,
        best_loss,
        keypoint_state,
        camera_state,
        parametrization_to_keypoints,
        optimizer,
        loss_function,
    )


@tf.function(experimental_relax_shapes=True)
def _train(
    steps: int,
    keypoint_params: tf.Variable,
    best_keypoint_params: tf.Variable,
    best_loss: tf.Variable,
    keypoint_state: tf.Tensor,
    camera_state: tf.Tensor,
    parametrization_to_keypoints: Callable,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_function: Callable,
) -> tf.Tensor:
    @tf.function
    def train_step() -> None:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(keypoint_params)

            estimated_keypoints = parametrization_to_keypoints(keypoint_params)

            loss_value = loss_function(
                estimated_keypoints=estimated_keypoints,
                keypoint_state=keypoint_state,
                camera_state=camera_state,
            )
        if loss_value < best_loss:
            best_loss.assign(loss_value)
            best_keypoint_params.assign(keypoint_params)
        optimizer.minimize(loss_value, keypoint_params, tape=tape)

    for _ in range(steps):
        train_step()

    return parametrization_to_keypoints(best_keypoint_params)


def _visualise(keypoint_state: np.ndarray) -> None:
    # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.set_xlim([0, 512])
    axis.set_ylim([0, 512])

    for kpt in keypoint_state:
        screen = Polygon(
            kpt,
            fc=(*tuple(np.random.rand(3)), 0.5),
            ec=(0, 0, 0, 1),
            lw=1,
        )
        axis.add_patch(screen)
        axis.autoscale_view()
    plt.savefig("generated_data.png")


def main() -> None:
    with open("../../tests/random_projections_and_keypoints.npy", "rb") as file:
        keypoints_in_3d = tf.constant(np.load(file), dtype=tf.float32)
        camera_state = tf.constant(np.load(file), dtype=tf.float32)
        keypoint_state = tf.constant(np.load(file), dtype=tf.float32)

    initial_estimate_params = encode_8_params(keypoints_in_3d)

    # _visualise(keypoint_state)

    train(
        steps=1,
        keypoint_state=keypoint_state,
        camera_state=camera_state,
        initial_estimate_params=initial_estimate_params,
        parametrization_to_keypoints=decode_8_params,
        optimizer=tf.keras.optimizers.SGD(learning_rate=6),
        loss_function=squared_reprojection_error,
    )


if __name__ == "__main__":
    main()

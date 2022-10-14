import argparse
from pathlib import Path

import tensorflow as tf
from matplotlib import pyplot as plt

from stereo.evaluations.data import serve_data_with_error
from stereo.evaluations.utils import read_keypoints
from stereo.implementations import PARAMETRIZATIONS, RectangleParametrization
from stereo.implementations.custom_optimization_procedure import (
    angular_reprojection_error,
    squared_reprojection_error,
)


# pylint: disable=invalid-name
def cartesian_product(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    a, b = a[None, :, None], b[:, None, None]
    return tf.concat([a + tf.zeros_like(b), tf.zeros_like(a) + b], 2)


def reprojection_error_wrapper(
    inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor]
) -> tf.Tensor:
    return squared_reprojection_error(*inputs)


def angular_reprojection_error_wrapper(
    inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor]
) -> tf.Tensor:
    return angular_reprojection_error(*inputs)


def get_orthogonal_vectors(dim: int) -> tuple[tf.Tensor, tf.Tensor]:
    eta, _ = tf.linalg.normalize(tf.random.normal((dim,), 1))
    delta, _ = tf.linalg.normalize(tf.random.normal((dim,), 1))
    delta, _ = tf.linalg.normalize(delta - tf.tensordot(eta, delta, axes=1) * eta)
    return eta, delta


def get_data_w_error(
    input_file: str, sigma: float
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    rectangles, camera_states, keypoint_states = read_keypoints(str(input_file))
    data_w_error = list(
        serve_data_with_error(sigma, 1, rectangles, keypoint_states, camera_states, 10)
    )[0]
    return data_w_error


def prepare_grid(
    parametrization: RectangleParametrization,
    rectangle: tf.Tensor,
    samples_per_axis: int,
) -> tf.Tensor:
    param_dim = {"8_params": 8, "9_params": 9, "rigid_body": 8, "rigid_body_quat": 9}[
        parametrization.name
    ]
    eta, delta = get_orthogonal_vectors(param_dim)

    origin = parametrization.encode(rectangle)
    epsilon_grid = tf.linspace(-10.0, 10.0, samples_per_axis)
    delta_grid = tf.linspace(-10.0, 10.0, samples_per_axis)

    return origin + tf.reduce_sum(
        cartesian_product(epsilon_grid, delta_grid)[..., tf.newaxis] * [eta, delta],
        axis=-2,
    )


def calculate_grid_loss(
    parametrization: RectangleParametrization,
    log_loss: bool,
    angular: bool,
    grid: tf.Tensor,
    keypoint_state: tf.Tensor,
    camera_state: tf.Tensor,
) -> tf.Tensor:
    samples_per_axis = tf.shape(grid)[0]
    samples = samples_per_axis**2

    squashed_grid = tf.reshape(grid, [-1, tf.shape(grid)[-1]])
    squashed_rects = tf.vectorized_map(parametrization.decode, squashed_grid)
    squashed_losses = tf.map_fn(
        angular_reprojection_error_wrapper if angular else reprojection_error_wrapper,
        (
            squashed_rects,
            tf.tile(keypoint_state[tf.newaxis, ...], [samples, 1, 1, 1]),
            tf.tile(camera_state[tf.newaxis, ...], [samples, 1, 1, 1]),
        ),
        dtype=(tf.float32, tf.float32, tf.float32),
        fn_output_signature=tf.float32,
    )
    if log_loss:
        squashed_losses = tf.math.log(squashed_losses)
    return tf.reshape(squashed_losses, [samples_per_axis, samples_per_axis])


def plot_grid(param: str, log_loss: bool, angular: bool, grid: tf.Tensor) -> None:
    plt.figure()
    plt.title(log_loss * "log " + angular * "angular " + f"loss for {param}")
    plot = plt.imshow(grid)
    plt.colorbar(plot)
    plt.show()


def main(
    input_file: Path,
    samples_per_axis: int,
    sigma: float,
    param: str,
    log_loss: bool,
    angular: bool,
) -> None:
    parametrization = PARAMETRIZATIONS[param]
    rectangle, keypoint_state, camera_state = get_data_w_error(str(input_file), sigma)
    grid = prepare_grid(parametrization, rectangle, samples_per_axis)
    loss_grid = calculate_grid_loss(
        parametrization, log_loss, angular, grid, keypoint_state, camera_state
    )
    plot_grid(param, log_loss, angular, loss_grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw the loss landscape for a random rectangle"
    )
    parser.add_argument(
        "--angular",
        action=argparse.BooleanOptionalAction,
        help="Use angular loss",
        default=False,
    )
    parser.add_argument(
        "--log_loss",
        action=argparse.BooleanOptionalAction,
        help="Plot log(loss)",
        default=False,
    )
    parser.add_argument(
        "--sigma",
        type=float,
        help="Standard deviation of recorded points",
        required=True,
    )
    parser.add_argument("--samples_per_axis", type=int, default=250)
    parser.add_argument(
        "--input", type=Path, help="Database of rectangles to use", required=True
    )
    parser.add_argument(
        "--param",
        type=str,
        choices=["8_params", "9_params", "rigid_body", "rigid_body_quat"],
        required=True,
    )
    args = parser.parse_args()
    main(
        args.input,
        args.samples_per_axis,
        args.sigma,
        args.param,
        args.log_loss,
        args.angular,
    )

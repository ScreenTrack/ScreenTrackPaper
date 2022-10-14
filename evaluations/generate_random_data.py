from __future__ import annotations

import argparse
import itertools
import math as m
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from tensorflow_graphics.geometry.transformation import quaternion
from tqdm import tqdm


@tf.function
def random_rectangles(
    count: tf.Tensor,
    coordinate_range: tf.Tensor,
    width_range: tf.Tensor,
    height_range: tf.Tensor,
) -> tf.Tensor:
    # pylint: disable=no-value-for-parameter
    position = tf.random.uniform((count, 3), -coordinate_range, coordinate_range)
    width = tf.random.uniform((count,), width_range[0], width_range[1])
    height = tf.random.uniform((count,), height_range[0], height_range[1])
    scale = tf.stack([width / 2, height / 2, tf.ones(count)], axis=1)
    rotation_quat = quaternion.from_euler(tf.random.uniform((count, 3), 0, 2 * m.pi))

    untransformed_rectangles = tf.tile(
        tf.constant(
            [
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                ],
            ]
        ),
        multiples=[count, 1, 1],
    )
    rectangles = (
        quaternion.rotate(
            untransformed_rectangles * scale[:, tf.newaxis, :],
            rotation_quat[:, tf.newaxis],
        )
    ) + position[:, tf.newaxis, :]
    return rectangles


@tf.function
def apply_cameras(
    projection_matrices: tf.Tensor,  # (B, 3, 4)
    points: tf.Tensor,  # (4, 3)
) -> tf.Tensor:
    # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
    points = tf.concat([tf.transpose(points, [1, 0]), tf.ones((1, 4))], axis=0)
    image_points = tf.transpose(
        projection_matrices @ points[tf.newaxis, ...], [0, 2, 1]
    )
    image_points, div = tf.split(image_points, [2, 1], -1)
    return image_points / div


@tf.function
def generate_camera_matrices(
    f_value: tf.Tensor,
    image_width: tf.Tensor,
    image_height: tf.Tensor,
    rotation_matrices: tf.Tensor,
    positions: tf.Tensor,
) -> tf.Tensor:
    intrinsic_matrix = tf.convert_to_tensor(
        [
            [f_value, 0.0, image_width / 2.0],
            [0.0, f_value, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    return (
        intrinsic_matrix[tf.newaxis, ...]
        @ rotation_matrices
        @ tf.concat(
            [
                tf.eye(3, batch_shape=tf.shape(rotation_matrices)[:1]),
                -positions[..., tf.newaxis],
            ],
            axis=-1,
        )
    )


@tf.function
def random_xy_rotations(z_axes: tf.Tensor) -> tf.Tensor:
    x_axes = tf.random.normal(shape=tf.shape(z_axes))
    x_axes, _ = tf.linalg.normalize(
        x_axes - tf.reduce_sum(x_axes * z_axes, axis=-1)[..., tf.newaxis] * z_axes,
        axis=-1,
    )
    y_axes = tf.linalg.cross(z_axes, x_axes)
    rotation_matrices = tf.stack([x_axes, y_axes, z_axes], axis=-2)
    return rotation_matrices


@tf.function
def rectangle_width_height(rectangle: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    width = rectangle[1] - rectangle[0]
    height = rectangle[3] - rectangle[0]
    return width, height


@tf.function
def rectangle_normal(rectangle: tf.Tensor) -> tf.Tensor:
    width, height = rectangle_width_height(rectangle)
    normal, _ = tf.linalg.normalize(tf.linalg.cross(width, height))
    return normal


@tf.function
def random_points_on_rectangle(
    rectangle: tf.Tensor, batch_size: tf.Tensor
) -> tf.Tensor:
    width, height = rectangle_width_height(rectangle)
    coord = tf.random.uniform((batch_size, 2), 0, 1)
    return rectangle[0] + coord @ [width, height]


@tf.function
def random_camera_positions(
    looking_ats: tf.Tensor,
    camera_distance_range: tf.Tensor,
) -> tf.Tensor:
    camera_distances = tf.random.uniform(
        (tf.shape(looking_ats)[0], 1),
        camera_distance_range[0],
        camera_distance_range[1],
    )
    camera_directions, _ = tf.linalg.normalize(
        tf.random.normal(shape=tf.shape(looking_ats)), axis=-1
    )
    camera_pos = looking_ats + camera_distances * camera_directions
    return camera_pos


@tf.function
def unsafe_random_views(
    rectangle: tf.Tensor,  # (4, 3)
    batch_size: tf.Tensor,  # ()
    camera_distance_range: tf.Tensor,  # (2,)
    f_value: tf.Tensor,  # ()
    image_width: tf.Tensor,  # ()
    image_height: tf.Tensor,  # ()
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    #  pick points to look at
    looking_ats = random_points_on_rectangle(rectangle, batch_size)

    #  pick positions for the camera looking at the chosen display point
    camera_positions = random_camera_positions(looking_ats, camera_distance_range)
    z_axes, _ = tf.linalg.normalize(looking_ats - camera_positions, axis=-1)

    #  pick camera rotations
    rotation_matrices = random_xy_rotations(z_axes)

    projection_matrices = generate_camera_matrices(
        f_value, image_width, image_height, rotation_matrices, camera_positions
    )
    image_rect = apply_cameras(projection_matrices, rectangle)

    return camera_positions, z_axes, projection_matrices, tf.cast(image_rect, tf.int32)


@tf.function
def random_sphere_points(
    center: tf.Tensor,  # (3,)
    radius: tf.Tensor,  # ()
    batch_size: tf.Tensor,  # ()
) -> tf.Tensor:
    pts = tf.random.normal((batch_size, 3))
    pts, _ = tf.linalg.normalize(pts, axis=-1)
    radius_3 = tf.random.uniform((batch_size,), 0, radius)
    pts = pts * tf.pow(radius_3, 1 / 3)[:, None]
    return center + pts


@tf.function
def unsafe_random_views_from_sphere(
    rectangle: tf.Tensor,  # (4, 3)
    center: tf.Tensor,  # (3,)
    radius: tf.Tensor,  # ()
    batch_size: tf.Tensor,  # ()
    f_value: tf.Tensor,  # ()
    image_width: tf.Tensor,  # ()
    image_height: tf.Tensor,  # ()
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    #  pick points to look at
    looking_ats = random_points_on_rectangle(rectangle, batch_size)

    #  pick positions for the camera looking at the chosen display point
    camera_positions = random_sphere_points(center, radius, batch_size)
    z_axes, _ = tf.linalg.normalize(looking_ats - camera_positions, axis=-1)

    #  pick camera rotations
    rotation_matrices = random_xy_rotations(z_axes)

    projection_matrices = generate_camera_matrices(
        f_value, image_width, image_height, rotation_matrices, camera_positions
    )
    image_rect = apply_cameras(projection_matrices, rectangle)

    return camera_positions, z_axes, projection_matrices, tf.cast(image_rect, tf.int32)


@tf.function
def rectangle_image_areas(image_rects: tf.Tensor) -> tf.Tensor:
    x_coordinates = image_rects[..., 0]
    y_coordinates = image_rects[..., 1]
    double_area = tf.math.abs(
        tf.reduce_sum(x_coordinates * tf.roll(y_coordinates, 1, -1), axis=-1)
        - tf.reduce_sum(tf.roll(x_coordinates, 1, -1) * y_coordinates, axis=-1)
    )
    return tf.cast(double_area, tf.float32) / 2.0


@tf.function
def are_view_rotations_valid(image_rects: tf.Tensor) -> tf.Tensor:
    return tf.math.reduce_all(
        [
            image_rects[:, 0, 0] < image_rects[:, 1, 0],
            image_rects[:, 1, 1] < image_rects[:, 2, 1],
            image_rects[:, 2, 0] > image_rects[:, 3, 0],
            image_rects[:, 3, 1] > image_rects[:, 0, 1],
        ],
        axis=0,
    )


@tf.function
def are_view_visibilities_valid(
    image_rects: tf.Tensor, image_width: tf.Tensor, image_height: tf.Tensor
) -> tf.Tensor:
    x_coordinates = image_rects[..., 0]
    y_coordinates = image_rects[..., 1]
    return tf.math.reduce_all(
        [
            tf.math.reduce_all(tf.less(x_coordinates, image_width), axis=-1),
            tf.math.reduce_all(tf.greater(x_coordinates, 0), axis=-1),
            tf.math.reduce_all(tf.less(y_coordinates, image_height), axis=-1),
            tf.math.reduce_all(tf.greater(y_coordinates, 0), axis=-1),
        ],
        axis=0,
    )


@tf.function
def is_rectangle_in_front(
    rectangle: tf.Tensor,  # (4, 3)
    positions: tf.Tensor,  # (B, 3)
    z_axes: tf.Tensor,  # (B, 3)
) -> tf.Tensor:  # (B,)
    # (B, 4, 3)
    towards_corners = rectangle[tf.newaxis, ...] - positions[:, tf.newaxis, ...]
    towards_corners, _ = tf.linalg.normalize(towards_corners)

    # (B, 4)
    dot_products = tf.reduce_sum(towards_corners * z_axes[:, tf.newaxis, :], axis=-1)
    return tf.reduce_all(dot_products > 0, axis=-1)


@tf.function
def are_views_valid(
    rectangle: tf.Tensor,
    positions: tf.Tensor,
    z_axes: tf.Tensor,
    image_rects: tf.Tensor,
    image_width: tf.Tensor,
    image_height: tf.Tensor,
    min_screen_percentage: tf.Tensor,
    max_viewing_angle: float,
) -> tf.Tensor:
    normal = rectangle_normal(rectangle)
    viewing_angles = tf.acos(tf.tensordot(z_axes, normal, [[1], [0]]))
    image_area = image_width * image_height
    return tf.math.reduce_all(
        [
            viewing_angles <= max_viewing_angle / 360 * 2 * m.pi,
            is_rectangle_in_front(rectangle, positions, z_axes),
            are_view_visibilities_valid(image_rects, image_width, image_height),
            are_view_rotations_valid(image_rects),
            rectangle_image_areas(image_rects) * 100
            >= image_area * min_screen_percentage,
        ],
        axis=0,
    )


def attempt_generation(
    rectangle: tf.Tensor,
    config: dict,
    generating_fn: Callable,
    generating_args: list,
) -> tuple[tf.Tensor | None, tf.Tensor | None]:
    projection_matrices = None
    image_rects = None

    image_width = config["image"]["width"]
    image_height = config["image"]["height"]
    (
        unsafe_positions,
        unsafe_z_axes,
        unsafe_projection_matrices,
        unsafe_image_rects,
    ) = generating_fn(*generating_args)
    valid = are_views_valid(
        rectangle,
        unsafe_positions,
        unsafe_z_axes,
        unsafe_image_rects,
        image_width,
        image_height,
        config["min_screen_percentage"],
        config["max_viewing_angle"],
    )
    if tf.reduce_any(valid):
        valid_idx = tf.squeeze(tf.where(valid), axis=-1)
        projection_matrices = tf.gather(unsafe_projection_matrices, valid_idx, axis=0)
        image_rects = tf.gather(unsafe_image_rects, valid_idx, axis=0)
    return projection_matrices, image_rects


def random_views(
    config: dict, count: int, rectangle: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    found = 0
    projection_matrices = []
    image_rects = []
    # pylint: disable=while-used
    while found < count:
        maybe_projection_matrices, maybe_image_rects = attempt_generation(
            rectangle,
            config,
            unsafe_random_views,
            [
                rectangle,
                config["batch_size"],
                config["camera_distance_range"],
                config["f_value"],
                config["image"]["width"],
                config["image"]["height"],
            ],
        )
        if maybe_projection_matrices is not None:
            found += len(maybe_projection_matrices)
            projection_matrices.append(maybe_projection_matrices)
            image_rects.append(maybe_image_rects)
    return (
        tf.concat(projection_matrices, 0)[:count, ...],
        tf.concat(image_rects, 0)[:count, ...],
    )


def random_views_from_sphere(
    config: dict, count: int, rectangle: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    # pylint: disable=while-used
    while True:
        looking_at = random_points_on_rectangle(rectangle, 1)
        center = random_camera_positions(looking_at, config["camera_distance_range"])[0]
        shortest_rectangle_side = tf.minimum(
            tf.norm(rectangle[0] - rectangle[1]), tf.norm(rectangle[0] - rectangle[3])
        )
        radius = (
            tf.random.uniform((), 0, 10)
            * shortest_rectangle_side
            / tf.norm(looking_at - center)
        )

        found = 0
        projection_matrices = []
        image_rects = []
        for iteration in itertools.count(0):
            maybe_projection_matrices, maybe_image_rects = attempt_generation(
                rectangle,
                config,
                unsafe_random_views_from_sphere,
                [
                    rectangle,
                    center,
                    radius,
                    config["batch_size"],
                    config["f_value"],
                    config["image"]["width"],
                    config["image"]["height"],
                ],
            )
            if maybe_projection_matrices is not None:
                found += len(maybe_projection_matrices)
                projection_matrices.append(maybe_projection_matrices)
                image_rects.append(maybe_image_rects)
            if (iteration == 1 and not found) or found > count:
                break
        if found > count:
            return (
                tf.concat(projection_matrices, 0)[:count, ...],
                tf.concat(image_rects, 0)[:count, ...],
            )


def save_to_file(
    filename: str,
    rectangles: tf.Tensor,
    cameras: tf.Tensor,
    keypoints: tf.Tensor,
) -> None:
    with open(filename, "wb") as file:
        np.save(file, rectangles.numpy())
        np.save(file, cameras.numpy())
        np.save(file, keypoints.numpy())


def main(config_file: Path) -> None:
    with config_file.open("rb") as file:
        config = yaml.load(file, Loader=yaml.CFullLoader)

    rectangles = random_rectangles(
        config["rectangle_count"],
        config["coordinate_range"],
        config["width_range"],
        config["height_range"],
    )
    cameras = []
    keypoints = []
    random_views_fn = random_views_from_sphere if config["sphere"] else random_views
    for rectangle in tqdm(rectangles):
        views_projection_matrices, views_keypoints = random_views_fn(
            config, config["views_per_rectangle"], rectangle
        )
        cameras.append(views_projection_matrices)
        keypoints.append(views_keypoints)
    cameras = tf.convert_to_tensor(cameras)
    keypoints = tf.convert_to_tensor(keypoints)
    save_to_file(config["output_path"], rectangles, cameras, keypoints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random rectangle view generator")
    parser.add_argument(
        "--config_file",
        type=Path,
        help="Path to the configuration file for rectangle generation",
        required=True,
    )
    args = parser.parse_args()
    main(args.config_file)

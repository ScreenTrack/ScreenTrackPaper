import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from loess.loess_1d import loess_1d
from tqdm import tqdm

from stereo.evaluations.data import introduce_error
from stereo.evaluations.evaluate_forward_error import _load_config, _parse_methods
from stereo.evaluations.utils import implementation_from_config, read_keypoints
from stereo.implementations.custom_optimization_procedure import get_rays_batched


@tf.function(
    input_signature=[
        (
            tf.TensorSpec((None, 4, 2), tf.float32),
            tf.TensorSpec((None, 3, 4), tf.float32),
        )
    ]
)
def get_dispersion(inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    keypoint_state, camera_state = inputs
    _, rays = get_rays_batched(keypoint_state, camera_state)  # (T, 4, 3)
    rays, _ = tf.linalg.normalize(rays, axis=-1)  # (T, 4, 3)
    return tf.reduce_min(1 - tf.linalg.norm(tf.reduce_mean(rays, axis=0), axis=-1))


def _maximum_3d_error(
    results: np.ndarray,  # (N, 4, 3)
    expected: np.ndarray,  # (N, 4, 3)
) -> tf.Tensor:
    height = tf.norm(expected[:, 3, :] - expected[:, 0, :], axis=-1)
    width = tf.norm(expected[:, 1, :] - expected[:, 0, :], axis=-1)
    shortest_side_length = tf.minimum(width, height)
    return tf.reduce_max(
        tf.norm(results - expected, axis=-1) / shortest_side_length[:, tf.newaxis],
        axis=-1,
    )


# pylint: disable=too-many-locals
def main(dataset_path: Path, config_path: Path, output_path: Path) -> None:
    rectangles, camera_states, keypoint_states = read_keypoints(str(dataset_path))
    config = _load_config(config_path)
    methods = _parse_methods(config["methods"])

    r_values = tf.vectorized_map(get_dispersion, [keypoint_states, camera_states])
    print(tf.reduce_mean(r_values), tf.math.reduce_std(r_values))
    keypoint_states_with_error = tf.convert_to_tensor(
        [introduce_error(kps, 0.1) for kps in keypoint_states]
    )
    for name, config in tqdm(methods.items()):
        if name == 'None':
            name = 'Baseline'
        implementation = implementation_from_config(config)
        results = tf.convert_to_tensor(
            [
                implementation(kps, cms)
                for kps, cms in zip(keypoint_states_with_error, camera_states)
            ]
        )
        error_values = 100 * _maximum_3d_error(results=results, expected=rectangles.numpy()).numpy()
        # pylint: disable=unbalanced-tuple-unpacking
        loessed_x, loessed_y, _ = loess_1d(
            r_values.numpy(),
            error_values,
            xnew=np.linspace(0, 0.2, 100),
            degree=1,
            frac=0.2,
        )
        plt.plot(loessed_x, loessed_y, label=name)
        plt.scatter(r_values, error_values, s=1/5000, marker=',', alpha=.24)

    plt.xlim([0, 0.2])
    plt.ylim([0, 70])
    plt.xlabel("View diversity ($R$)")
    plt.ylabel("Screen reconstruction error ($E_{sr}$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path / "smoothed_diversity.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot graph of dependence on view dispersion"
    )
    parser.add_argument("--style_file", type=Path, required=True)
    parser.add_argument("--config_file", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plt.style.use(["science", "ieee", args.style_file])
    main(args.dataset, args.config_file, args.output)

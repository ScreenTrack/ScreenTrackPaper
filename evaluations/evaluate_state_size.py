import argparse
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from stereo.evaluations.data import serve_data_with_error
from stereo.evaluations.evaluate_forward_error import (
    _load_config,
    _maximum_3d_error,
    _parse_methods,
)
from stereo.evaluations.utils import read_keypoints
from stereo.implementations import IMPLEMENTATIONS


def run_methods(methods: dict, samples: list, pbar: tqdm = None) -> tuple:
    results = {}
    times = {}
    rectangles: np.ndarray = np.array([sample[0] for sample in samples])
    for name, config in methods.items():
        config = config.copy()
        implementation = IMPLEMENTATIONS[config.pop("implementation")]
        if "state_length" in config:
            config.pop("state_length")
        if config:
            implementation.set_hyperparameters(**config)
        if pbar is not None:
            pbar.set_description(f"Compiling {name}")
        implementation(*samples[0][1:])
        if pbar is not None:
            pbar.set_description(f"Running {name}")
        start = time.process_time_ns()
        estimated: np.ndarray = np.array(
            [implementation(*sample[1:]) for sample in samples]
        )
        end = time.process_time_ns()
        results[name] = _maximum_3d_error(results=estimated, expected=rectangles) / len(
            rectangles
        )
        times[name] = (end - start) / len(rectangles)
    return results, times


def draw_plots(
    state_lengths: np.ndarray, errors: dict, times: dict, output_path: Path
) -> None:
    fig = plt.figure()

    times_plt = fig.add_subplot(2, 1, 1)
    for method, y_values in times.items():
        if method == 'None':
            method = 'Baseline'
        times_plt.plot(state_lengths, np.array(y_values) / 1e6, label=method)
    plt.xticks(np.arange(2, 21, 2))
    times_plt.set_ylabel("Execution time ($t/\\textrm{ms}$)")
    plt.legend()

    err_3d = fig.add_subplot(2, 1, 2)
    for method, y_values in errors.items():
        if method == 'None':
            method = 'Baseline'
        err_3d.plot(state_lengths, 100 * np.array(y_values), label=method)
    plt.xticks(np.arange(2, 21, 2))
    err_3d.set_ylabel("Screen reconstruction error ($E_{sr}$)")

    plt.xlabel("Number of views ($k$)")
    fig.tight_layout()
    plt.savefig(output_path)


# pylint: disable=too-many-locals
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style_file", type=Path, required=True)
    parser.add_argument("--config_file", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rectangles, camera_states, keypoint_states = read_keypoints(str(args.dataset))
    config = _load_config(args.config_file)
    methods = _parse_methods(config["methods"])
    # pre-compile
    run_methods(
        methods,
        list(
            serve_data_with_error(
                args.sigma,
                config["trials"],
                rectangles,
                keypoint_states,
                camera_states,
                2,
            )
        ),
    )
    state_lengths = np.arange(2, 21)
    time_durations: dict[str, list] = {method: [] for method in methods.keys()}
    error_values: dict[str, list] = {method: [] for method in methods.keys()}
    for state_length in (pbar := tqdm(state_lengths)):
        pbar.set_description(f"Introducing error state_length = {state_length}")
        data_with_error = list(
            serve_data_with_error(
                args.sigma,
                config["trials"],
                rectangles,
                keypoint_states,
                camera_states,
                state_length,
            )
        )
        errors, times = run_methods(methods, data_with_error, pbar)
        for method in methods.keys():
            time_durations[method].append(times[method])
            error_values[method].append(errors[method])

    plt.style.use(["science", "ieee", args.style_file])
    draw_plots(
        state_lengths,
        error_values,
        time_durations,
        args.output / f"state_size_{config['output_name']}.pdf",
    )


if __name__ == "__main__":
    main()

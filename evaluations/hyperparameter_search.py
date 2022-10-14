import argparse
import pickle
from pathlib import Path

import optuna
import tensorflow as tf

from stereo.evaluations.data import serve_data_with_error
from stereo.evaluations.evaluate_forward_error import _maximum_3d_error
from stereo.evaluations.utils import read_keypoints
from stereo.implementations import IMPLEMENTATIONS, LOSSES
from stereo.implementations.custom_optimization_procedure import _train
from stereo.implementations.levenberg_marquardt import parametrized_levenberg_marquardt

TRIALS = 1000
STATE_LENGTH = 10
STEPS = 100


# pylint: disable=protected-access
def cleanup_graphs(tf_function: tf.types.experimental.GenericFunction) -> None:
    if tf_function._stateful_fn is not None:
        tf_function._stateful_fn._function_cache.clear()
    if tf_function._stateless_fn is not None:
        tf_function._stateless_fn._function_cache.clear()


def objective(trial: optuna.Trial) -> float:
    implementation = None
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "lma"])

    if optimizer == "adam":
        implementation = IMPLEMENTATIONS[IMPLEMENTATION]
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        beta_1 = trial.suggest_float("beta_1", 0, 1)
        beta_2 = trial.suggest_float("beta_2", 0, 1)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
        )
        implementation.set_hyperparameters(
            optimizer=optimizer, steps=STEPS, loss_function=LOSSES["squared"]
        )
    elif optimizer == "sgd":
        implementation = IMPLEMENTATIONS[IMPLEMENTATION]
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        momentum = trial.suggest_float("momentum", 0, 1)
        use_nesterov = trial.suggest_categorical("nesterov", [True, False])
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=use_nesterov,
        )
        implementation.set_hyperparameters(
            optimizer=optimizer, steps=STEPS, loss_function=LOSSES["squared"]
        )
    else:
        implementation = IMPLEMENTATIONS["LM-" + IMPLEMENTATION]
        regularizer = trial.suggest_float("regularizer", 1e-25, 1e-15, log=True)
        regularizer_multiplier = trial.suggest_float(
            "regularizer_multiplier", 1, 1e3, log=True
        )
        implementation.set_hyperparameters(
            steps=STEPS,
            regularizer=regularizer,
            regularizer_multiplier=regularizer_multiplier,
        )

    within_threshold = 0
    for rectangle, keypoint_state, camera_state in serve_data_with_error(
        SIGMA,
        TRIALS,
        RECTANGLES,
        FULL_KEYPOINT_STATES,
        FULL_CAMERA_STATES,
        STATE_LENGTH,
    ):
        estimate = implementation(keypoint_state, camera_state)
        within_threshold += _maximum_3d_error(estimate[None, ...], rectangle[None, ...])
    cleanup_graphs(_train)
    cleanup_graphs(parametrized_levenberg_marquardt)
    return within_threshold / TRIALS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize hyperparams")
    parser.add_argument("--dataset", type=Path, default=Path("dataset.npy"))
    parser.add_argument(
        "--implementation",
        type=str,
        choices=["8_params", "9_params", "rigid_body", "rigid_body_quat", "raw"],
        required=True,
    )
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--optuna_trials", type=int, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    args = parser.parse_args()

    #  todo clean up
    RECTANGLES, FULL_CAMERA_STATES, FULL_KEYPOINT_STATES = read_keypoints(
        str(args.dataset)
    )
    IMPLEMENTATION = args.implementation
    SIGMA = args.sigma
    OPTUNA_TRIALS = args.optuna_trials

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    print(study.best_params)
    with args.output_file.open("wb") as study_file:
        pickle.dump(study, study_file)

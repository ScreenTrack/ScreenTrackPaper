from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from stereo.implementations.levenberg_marquardt import levenberg_marquardt as lm
from stereo.implementations.levenberg_marquardt import parametrized_levenberg_marquardt
from stereo.implementations.levenberg_marquardt import (
    reconstruct_3d_inhomogenous as inhomo,
)

from .custom_optimization_procedure import (
    angular_reprojection_error,
    decode_8_params,
    decode_9_params,
    decode_rigid_body,
    decode_rigid_body_quat,
    encode_8_params,
    encode_9_params,
    encode_rigid_body,
    encode_rigid_body_quat,
    squared_reprojection_error,
    train,
)


@dataclass
class RectangleParametrization:
    name: str
    encode: Callable
    decode: Callable


PARAMETRIZATIONS = {
    "8_params": RectangleParametrization(
        name="8_params",
        encode=encode_8_params,
        decode=decode_8_params,
    ),
    "9_params": RectangleParametrization(
        name="9_params",
        encode=encode_9_params,
        decode=decode_9_params,
    ),
    "rigid_body": RectangleParametrization(
        name="rigid_body",
        encode=encode_rigid_body,
        decode=decode_rigid_body,
    ),
    "rigid_body_quat": RectangleParametrization(
        name="rigid_body_quat",
        encode=encode_rigid_body_quat,
        decode=decode_rigid_body_quat,
    ),
    "raw": RectangleParametrization(
        name="raw",
        encode=tf.identity,
        decode=tf.identity,
    ),
}


class Optimization:
    def __init__(
        self,
        optimization_procedure: Callable,
    ):
        self.optimization_procedure = optimization_procedure

    def set_hyperparameters(self, **kwargs: Any) -> None:
        pass

    def __call__(
        self,
        keypoint_state: tf.Tensor,
        camera_state: tf.Tensor,
    ) -> tf.Tensor:
        return self.optimization_procedure([keypoint_state, camera_state])


class InitializedOptimization(Optimization):
    def __init__(
        self,
        optimization_procedure: Callable,
        initializer: Callable = inhomo,
    ):
        super().__init__(optimization_procedure)
        self.initializer = initializer

    def __call__(
        self,
        keypoint_state: tf.Tensor,
        camera_state: tf.Tensor,
    ) -> tf.Tensor:
        return self.optimization_procedure(
            [
                self.initializer([keypoint_state, camera_state]),
                keypoint_state,
                camera_state,
            ]
        )


class CustomOptimization(Optimization):
    _hparams: dict[str, Any] = {}

    def __init__(
        self,
        optimization_procedure: Callable,
        parametrization: RectangleParametrization,
        initializer: Callable = inhomo,
    ):
        super().__init__(
            optimization_procedure=optimization_procedure,
        )
        self.parametrization = parametrization
        self.initializer = initializer

    def set_hyperparameters(self, **kwargs: Any) -> None:
        self._hparams = kwargs

    def __call__(
        self,
        keypoint_state: tf.Tensor,
        camera_state: tf.Tensor,
    ) -> tf.Tensor:
        initial_estimate = self.parametrization.encode(
            self.initializer([keypoint_state, camera_state]),
        )

        return self.optimization_procedure(
            keypoint_state=keypoint_state,
            camera_state=camera_state,
            initial_estimate_params=initial_estimate,
            parametrization_to_keypoints=self.parametrization.decode,
            **self._hparams,
        )


IMPLEMENTATIONS = {
    "nehomogena": Optimization(inhomo),
    "Levenberg-Marquardt": InitializedOptimization(lm),
    **{
        name: CustomOptimization(
            optimization_procedure=train,
            parametrization=parametrization,
        )
        for name, parametrization in PARAMETRIZATIONS.items()
    },
    **{
        f"LM-{name}": CustomOptimization(
            optimization_procedure=parametrized_levenberg_marquardt,
            parametrization=parametrization,
        )
        for name, parametrization in PARAMETRIZATIONS.items()
    },
}

LOSSES = {
    "squared": squared_reprojection_error,
    "angular": angular_reprojection_error,
}

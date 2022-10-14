import argparse
from collections.abc import Generator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stereo.evaluations.evaluate_on_rw import IMAGE_SHAPE, draw_quads
from stereo.evaluations.run_model_on_rw import image_paths


def resource_paths(scene_path: Path) -> Generator:
    for image_path in image_paths(scene_path):
        gt_path = (scene_path / "GT") / image_path.with_suffix(".npy").name
        yield image_path, gt_path


# pylint: disable=no-member, unused-argument
def click_event(
    event: Any, x_coordinate: int, y_coordinate: int, flags: Any, param: Any
) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    gt_idx, gt_locations, orig_img = param
    gt_locations[gt_idx[0]] = [x_coordinate, IMAGE_SHAPE[1] - y_coordinate]
    gt_idx[0] = (gt_idx[0] + 1) % 4
    if not gt_idx[0]:
        annotated_img = orig_img.copy()
        draw_quads(annotated_img, gt_locations, (255, 0, 0))
        cv2.imshow("Data annotator", annotated_img)


def main(scene_path: Path) -> None:
    for image_path, gt_path in resource_paths(scene_path):
        gt_idx: np.ndarray = np.zeros((2,), dtype=np.int32)
        gt_locations: np.ndarray = np.zeros((4, 2), dtype=np.float32)
        if gt_path.exists():
            gt_locations = np.load(gt_path) * IMAGE_SHAPE
        orig_img = cv2.imread(str(image_path))

        annotated_img = orig_img.copy()
        draw_quads(annotated_img, gt_locations, (255, 0, 0))
        cv2.imshow("Data annotator", annotated_img)
        cv2.setMouseCallback(
            "Data annotator", click_event, (gt_idx, gt_locations, orig_img)
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        np.save(gt_path, gt_locations / IMAGE_SHAPE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        "-i",
        help="Path to the scene to process.",
        type=Path,
        dest="scene_path",
        required=True,
    )
    args = parser.parse_args()
    main(args.scene_path)

import re

import cv2
import numpy as np
from fastapi import FastAPI
from starlette.responses import StreamingResponse

import src.globals as g
import supervisely as sly
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.cuboid import Cuboid
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging.color import _validate_hex_color, hex2rgb, random_rgb, rgb2hex


def get_thickness(render: np.ndarray, thickness_percent: float) -> int:
    render_height, render_width, _ = render.shape
    return int(render_width * thickness_percent / 100)


def get_rgba_np(
    ann: sly.Annotation,
    OUTPUT_WIDTH_PX: int,
    BBOX_THICKNESS_PERCENT: float,
    BBOX_OPACITY: float,
    FILLBBOX_OPACITY: float,
    MASK_OPACITY: float,
    project_id: int,
    image_id: int,
) -> np.ndarray:
    try:
        out_size = (int((ann.img_size[0] / ann.img_size[1]) * OUTPUT_WIDTH_PX), OUTPUT_WIDTH_PX)
        ann = ann.resize(out_size, skip_empty_masks=True)

        render_mask, render_bbox, render_fillbbox = (
            np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8),
            np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8),
            np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8),
        )

        for label in ann.labels:
            label: sly.Label
            if type(label.geometry) == sly.Point:
                label.draw(render_mask, thickness=25)
            elif type(label.geometry) == sly.Rectangle:
                thickness = get_thickness(render_bbox, BBOX_THICKNESS_PERCENT)
                label.draw_contour(render_bbox, thickness=thickness)
                label.draw(render_fillbbox)
            else:
                label.draw(render_mask)

        alpha_mask = (
            MASK_OPACITY - np.all(render_mask == [0, 0, 0], axis=-1).astype("uint8")
        ) * 255
        alpha_mask[alpha_mask < 0] = 0

        alpha_bbox = (
            BBOX_OPACITY - np.all(render_bbox == [0, 0, 0], axis=-1).astype("uint8")
        ) * 255
        alpha_bbox[alpha_bbox < 0] = 0

        alpha_fillbbox = (
            FILLBBOX_OPACITY - np.all(render_fillbbox == [0, 0, 0], axis=-1).astype("uint8")
        ) * 255
        alpha_fillbbox[alpha_fillbbox < 0] = 0

        alpha = np.where(alpha_mask != 0, alpha_mask, alpha_fillbbox)
        alpha = np.where(alpha_bbox != 0, alpha_bbox, alpha)

        rgba_mask = np.dstack((render_mask, alpha_mask))
        rgba_bbox = np.dstack((render_bbox, alpha_bbox))
        render_fillbbox = np.dstack((render_fillbbox, alpha_fillbbox))

        rgba = np.where(rgba_mask != 0, rgba_mask, render_fillbbox)
        rgba = np.where(rgba_bbox != 0, rgba_bbox, rgba)

    except Exception as e:
        new_error_message = f"PROJECT ID: {project_id}, IMAGE ID: {image_id}. Error: {e}"
        raise e.__class__(new_error_message) from e

    return rgba, alpha, out_size


def handle_broken_annotations(jann, json_project_meta):
    ann_ids = [obj["classId"] for obj in jann["objects"]]
    filtered_cls = [cls for cls in json_project_meta["classes"] if cls["id"] in ann_ids]

    sorted_ann = sorted(jann["objects"], key=lambda x: x["classId"])
    sorted_cls = sorted(filtered_cls, key=lambda x: x["id"])

    def _conditions(_ann, _cls):
        if (
            _ann["geometryType"] == Bitmap.geometry_name()
            and _cls["shape"] == Rectangle.geometry_name()
        ):
            return True
        if _ann["geometryType"] == Cuboid.geometry_name() and len(_ann["points"]) < 7:
            return True

        return False

    cls_to_drop = [
        _ann["classId"]
        for _ann, _cls in zip(sorted_ann, sorted_cls)
        if _conditions(_ann, _cls) is True
    ]
    return [obj for obj in jann["objects"] if obj["classId"] not in cls_to_drop]


def handle_broken_project_meta(json_project_meta: dict, e):
    if "Supported only HEX RGB string format!" in str(e):
        for idx, cls in enumerate(json_project_meta["classes"]):
            if _validate_hex_color(cls["color"]) is False:
                json_project_meta["classes"][idx]["color"] = rgb2hex(random_rgb())
    else:
        raise e

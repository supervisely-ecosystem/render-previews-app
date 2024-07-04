import os
import re

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Response
from starlette.responses import StreamingResponse

import src.globals as g
import supervisely as sly
from src.ui import get_settings
from supervisely import ImageInfo, ProjectInfo
from supervisely.annotation.tag import TagJsonFields
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.cuboid import Cuboid
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging.color import _validate_hex_color, hex2rgb, random_rgb, rgb2hex


def get_thickness(render: np.ndarray, thickness_percent: float, from_min=False) -> int:
    render_height, render_width, _ = render.shape
    render_side = render_width
    if from_min:
        render_side = min(render_height, render_width)
    return int(render_side * thickness_percent / 100)


def get_rendered_image(image_id, project_id, json_project_meta):
    try:
        project_meta = sly.ProjectMeta.from_json(json_project_meta)
    except Exception as e:  # Error: Supported only HEX RGB string format!
        json_project_meta = handle_broken_project_meta(json_project_meta)
        project_meta = sly.ProjectMeta.from_json(json_project_meta)

    try:
        jann = g.api.annotation.download_json(image_id)
    except requests.exceptions.HTTPError as e:
        sly.logger.error(str(e))  # image not accessed
        raise HTTPException(status_code=404, detail=str(e))

    try:
        try:
            ann = sly.Annotation.from_json(jann, project_meta)
        except ValueError as e:  # Tag Meta is none
            json_project_meta = g.api.project.get_meta(project_id)
            g.JSON_METAS[project_id] = json_project_meta
            project_meta = sly.ProjectMeta.from_json(json_project_meta)
            jann = g.api.annotation.download_json(image_id)
            ann = sly.Annotation.from_json(jann, project_meta)
        # except KeyError as e:  # missing fields in api response
        #     if e.args[0].value in TagJsonFields.values():
        #         tmp = jann.copy()
        #         tmp["tags"] = []
        #         ann = sly.Annotation.from_json(tmp, project_meta)

    except RuntimeError:
        # case 1: new class added to image, but meta is old
        json_project_meta = g.api.project.get_meta(project_id)
        g.JSON_METAS[project_id] = json_project_meta
        project_meta = sly.ProjectMeta.from_json(json_project_meta)
        try:
            ann = sly.Annotation.from_json(jann, project_meta)
        except RuntimeError:
            # case 2: broken annotations
            jann["objects"] = handle_broken_annotations(jann, json_project_meta)
            ann = sly.Annotation.from_json(jann, project_meta)

    if any([True for val in ann.img_size if val is None]):
        raise HTTPException(
            status_code=500,
            detail="The image file has no information about its size. Please check the integrity of your project.",
        )

    settings = get_settings()
    rgba, _, _ = get_rgba_np(
        ann,
        settings.get("OUTPUT_WIDTH_PX", 500),
        settings.get("BBOX_THICKNESS_PERCENT", 0.5),
        settings.get("BBOX_OPACITY", 1),
        settings.get("FILLBBOX_OPACITY", 0.2),
        settings.get("MASK_OPACITY", 0.7),
        project_id,
        image_id,
    )

    rgba = cv2.cvtColor(rgba.astype("uint8"), cv2.COLOR_RGBA2BGRA)
    return cv2.imencode(".png", rgba)


def get_rgba_np(
    ann: sly.Annotation,
    OUTPUT_WIDTH_PX: int,
    BBOX_THICKNESS_PERCENT: float,
    BBOX_OPACITY: float,
    FILLBBOX_OPACITY: float,
    MASK_OPACITY: float,
    project_id: int,
    image_id: int,
    draw_class_names: bool = None,
    draw_tags: bool = None,
    with_image=None,
    bitmap: np.ndarray = None,
    skip_resize=False,
) -> np.ndarray:
    try:
        if skip_resize:
            out_size = ann.img_size
        else:
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
                label.draw(
                    render_mask,
                    thickness=get_thickness(render_mask, thickness_percent=3, from_min=True),
                )
            elif type(label.geometry) in (sly.GraphNodes, sly.Polyline):
                label.draw(
                    render_mask,
                    thickness=get_thickness(render_mask, thickness_percent=2, from_min=True),
                )
            elif type(label.geometry) == sly.Rectangle:
                thickness = get_thickness(render_bbox, BBOX_THICKNESS_PERCENT)
                label.draw_contour(
                    render_bbox,
                    thickness=thickness,
                    # draw_tags=draw_tags, #TODO fix (0,0,0,255) color font
                    # draw_class_name=draw_class_names,
                )
                label.draw(render_fillbbox)
            else:
                label.draw(
                    render_mask,
                    # draw_tags=draw_tags,
                    # draw_class_name=draw_class_names,
                )

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

        rgb = np.where(render_mask != 0, render_mask, render_fillbbox)
        rgb = np.where(render_bbox != 0, render_bbox, rgb)

        rgba = np.dstack((rgb, alpha))

        result = np.zeros_like(rgb, dtype=np.uint8)
        if with_image is not None:
            if bitmap is not None:
                alpha_ = rgba[:, :, 3] / 255.0
                alpha_inv = 1.0 - alpha_
                # result = np.zeros_like(bitmap, dtype=np.uint8)
                for i in range(3):  # Loop over RGB channels
                    result[:, :, i] = (alpha_ * rgba[:, :, i] + alpha_inv * bitmap[:, :, i]).astype(
                        np.uint8
                    )
            else:
                alpha_ = rgba[:, :, 3] / 255.0
                for i in range(3):
                    result[:, :, i] = (alpha_ * rgba[:, :, i]).astype(np.uint8)

            for label in ann.labels:
                font = label._get_font(result.shape[:2])
                if draw_tags:
                    label._draw_tags(result, font, add_class_name=draw_class_names)
                elif draw_class_names:
                    label._draw_class_name(result, font)
        else:
            result = rgba

    except Exception as e:
        new_error_message = f"PROJECT ID: {project_id}, IMAGE ID: {image_id}. Error: {e}"
        raise e.__class__(new_error_message) from e

    return result, alpha, out_size


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
        if _ann["geometryType"] == Polygon.geometry_name() and len(_ann["points"]["exterior"]) < 3:
            return True
        if (
            _ann["geometryType"] == Rectangle.geometry_name()
            and len(_ann["points"]["exterior"]) < 2
        ):
            return True

        return False

    cls_to_drop = [
        _ann["classId"]
        for _ann, _cls in zip(sorted_ann, sorted_cls)
        if _conditions(_ann, _cls) is True
    ]
    return [obj for obj in jann["objects"] if obj["classId"] not in cls_to_drop]


def handle_broken_project_meta(json_project_meta: dict) -> dict:
    for idx, cls in enumerate(json_project_meta["classes"]):
        if _validate_hex_color(cls["color"]) is False:
            new_color = rgb2hex(random_rgb())
            sly.logger.warning(
                f"'{cls['color']}' is not validated as hex. Trying to convert it to: {new_color}"
            )
            json_project_meta["classes"][idx]["color"] = new_color
        # for edge in cls["geometry_config"]["edges"]:
        #     curr_color = edge.get("color")
        #     new_color = rgb2hex(random_rgb())
        #     if curr_color is None:
        #         edge["color"] = new_color
        #     else:
        #         if _validate_hex_color("#" + curr_color) is True:
        #             edge["color"] = "#" + edge["color"]
        #         if _validate_hex_color(edge["color"]) is False:
        #             sly.logger.warning(
        #                 f"'{cls['color']}' is not validated as hex. Trying to convert it to: {new_color}"
        #             )
        for node, data in cls["geometry_config"]["nodes"].items():
            curr_color = data.get("color")
            new_color = rgb2hex(random_rgb())
            if curr_color is None:
                data["color"] = new_color
            else:
                if _validate_hex_color("#" + curr_color) is True:
                    data["color"] = "#" + data["color"]
                if _validate_hex_color(data["color"]) is False:
                    sly.logger.warning(
                        f"'{cls['color']}' is not validated as hex. Trying to convert it to: {new_color}"
                    )

    return json_project_meta


def image_was_updated(project: ProjectInfo, image: ImageInfo) -> bool:

    cache_dir = f"{g.STORAGE_DIR}/cached_updated_at/{project.id}"
    os.makedirs(cache_dir, exist_ok=True)
    image_path = f"{cache_dir}/{image.id}.txt"

    if sly.fs.file_exists(image_path):
        with open(image_path, "r") as file:
            cached_updated_at = file.read()
        if cached_updated_at != image.updated_at:
            return True

    with open(image_path, "w") as file:
        file.write(image.updated_at)

    return False

from pathlib import Path
from typing import Literal, Union

import cv2
import requests
from fastapi import HTTPException, Response

import src.globals as g
import src.utils as u
import supervisely as sly
from src.ui import card_1, get_settings
from supervisely.app.widgets import Container

layout = Container(widgets=[card_1], direction="vertical")

static_dir = Path(g.STORAGE_DIR)
app = sly.Application(layout=layout, static_dir=static_dir)
server = app.get_server()


@server.get("/refresh")
def refresh_project_list():
    try:
        g.JSON_METAS = g.update_metas()
        return "Projects successfully refreshed"
    except Exception as e:
        return f"Error: {e}"


@server.get("/renders", response_class=Response)
def image_endpoint(project_id: int, image_id: int):
    try:
        json_project_meta = g.JSON_METAS[project_id]
    except (KeyError, TypeError):
        json_project_meta = g.api.project.get_meta(project_id)
        g.JSON_METAS[project_id] = json_project_meta

    try:
        try:
            project_meta = sly.ProjectMeta.from_json(json_project_meta)
        except ValueError as e:  # Error: Supported only HEX RGB string format!
            json_project_meta = u.handle_broken_project_meta(json_project_meta)
            project_meta = sly.ProjectMeta.from_json(json_project_meta)

        try:
            jann = g.api.annotation.download_json(image_id)
        except requests.exceptions.HTTPError as e:
            sly.logger.error(str(e))  # image not accessed
            raise HTTPException(status_code=404, detail=str(e))

        try:
            try:
                ann = sly.Annotation.from_json(jann, project_meta)
            except ValueError:  # Tag Meta is none
                json_project_meta = g.api.project.get_meta(project_id)
                g.JSON_METAS[project_id] = json_project_meta
                project_meta = sly.ProjectMeta.from_json(json_project_meta)
                jann = g.api.annotation.download_json(image_id)
                ann = sly.Annotation.from_json(jann, project_meta)

        except RuntimeError:
            # case 1: new class added to image, but meta is old
            json_project_meta = g.api.project.get_meta(project_id)
            g.JSON_METAS[project_id] = json_project_meta
            project_meta = sly.ProjectMeta.from_json(json_project_meta)
            try:
                ann = sly.Annotation.from_json(jann, project_meta)
            except RuntimeError:
                # case 2: broken annotations
                jann["objects"] = u.handle_broken_annotations(jann, json_project_meta)
                ann = sly.Annotation.from_json(jann, project_meta)

        if any([True for val in ann.img_size if val is None]):
            raise HTTPException(
                status_code=500,
                detail="The image file has no information about its size. Please check the integrity of your project.",
            )

        settings = get_settings()
        rgba, _, _ = u.get_rgba_np(
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
        success, im = cv2.imencode(".png", rgba)

    except HTTPException as e:
        new_error_message = f"PROJECT_ID: {project_id}, IMAGE_ID: {image_id}. Error: {e.detail}"
        raise HTTPException(status_code=500, detail=new_error_message)
    except Exception as e:
        new_error_message = f"PROJECT_ID: {project_id}, IMAGE_ID: {image_id}. Error: {str(e)}"
        raise e.__class__(new_error_message) from e

    headers = {"Cache-Control": "max-age=604800", "Content-Type": "image/png"}
    return Response(im.tobytes(), headers=headers, media_type="image/png")


@server.get("/render-on-image", response_class=Response)
def render_on_img_endpoint(
    image_id: int,
    classname: Literal["0", "1"] = "0",
    tags: Literal["0", "1"] = "0",
    onlyann: Literal["0", "1"] = "0",
):

    try:
        image_info = g.api.image.get_info_by_id(image_id)
        if image_info is None:
            raise ValueError(f"The image {image_id} is not existed.")

        dataset = g.api.dataset.get_info_by_id(image_info.dataset_id)
        json_project_meta = g.api.project.get_meta(dataset.project_id)
        project_meta = sly.ProjectMeta.from_json(json_project_meta)

        jann = g.api.annotation.download_json(image_id)
        ann = sly.Annotation.from_json(jann, project_meta)

        draw_class_names = True if classname == "1" else False
        draw_tags = True if tags == "1" else False
        with_image = True if onlyann == "0" else False
        np_image = g.api.image.download_np(image_id) if with_image else None

        settings = get_settings("render-on-image")
        rgba, _, _ = u.get_rgba_np(
            ann,
            settings.get("OUTPUT_WIDTH_PX", 500),
            settings.get("BBOX_THICKNESS_PERCENT", 0.5),
            settings.get("BBOX_OPACITY", 1),
            settings.get("FILLBBOX_OPACITY", 0.2),
            settings.get("MASK_OPACITY", 0.7),
            dataset.project_id,
            image_id,
            draw_class_names,
            draw_tags,
            with_image,
            np_image,
            skip_resize=True,
        )

        rgba = cv2.cvtColor(rgba.astype("uint8"), cv2.COLOR_RGBA2BGRA)
        success, im = cv2.imencode(".png", rgba)

    except Exception as e:
        new_error_message = (
            f"PROJECT_ID: {dataset.project_id}, IMAGE_ID: {image_id}. Error: {str(e)}"
        )
        raise e.__class__(new_error_message) from e

    headers = {"Cache-Control": "max-age=604800", "Content-Type": "image/png"}
    return Response(im.tobytes(), headers=headers, media_type="image/png")

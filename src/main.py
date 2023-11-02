from pathlib import Path

import cv2
from fastapi import Response

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
async def image_endpoint(project_id: int, image_id: int):
    try:
        project_meta = g.JSON_METAS[project_id]
    except TypeError:
        project_meta = g.api.project.get_meta(project_id)

    project_meta = sly.ProjectMeta.from_json(project_meta)
    jann = g.api.annotation.download_json(image_id)
    ann = sly.Annotation.from_json(jann, project_meta)

    settings = get_settings()
    rgba, _, _ = u.get_rgba_np(
        ann,
        settings.get("OUTPUT_WIDTH_PX", 500),
        settings.get("BBOX_THICKNESS_PERCENT", 0.5),
        settings.get("BBOX_OPACITY", 1),
        settings.get("FILLBBOX_OPACITY", 0.2),
        settings.get("MASK_OPACITY", 0.7),
    )

    rgba = cv2.cvtColor(rgba.astype("uint8"), cv2.COLOR_RGBA2BGRA)
    success, im = cv2.imencode(".png", rgba)

    headers = {"Cache-Control": "max-age=604800", "Content-Type": "image/png"}
    return Response(im.tobytes(), headers=headers, media_type="image/png")

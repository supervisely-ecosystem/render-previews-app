import io

import cv2

# import httpx
import numpy as np
import requests
from fastapi import FastAPI, Form, Response
from PIL import Image
from starlette.responses import StreamingResponse

import src.globals as g
import src.utils as u
import supervisely as sly

# from fastapi.responses import StreamingResponse


# from src.ui import settings_dict

settings_dict = {
    "BBOX_THICKNESS_PERCENT": 0.5,
    "FILLBBOX_OPACITY": 0.2,
    "MASK_OPACITY": 0.7,
    "OUTPUT_WIDTH_PX": 500,
}

app = FastAPI()


@app.get("/test")
def test():
    return "testt"


@app.get("/renders", response_class=Response)
async def image_endpoint(project_id: int, image_id: int):
    OUTPUT_WIDTH_PX = settings_dict.get("OUTPUT_WIDTH_PX", 500)
    BBOX_THICKNESS_PERCENT = settings_dict.get("BBOX_THICKNESS_PERCENT", 0.005)
    BBOX_OPACITY = 1
    FILLBBOX_OPACITY = settings_dict.get("FILLBBOX_OPACITY", 0.2)
    MASK_OPACITY = settings_dict.get("MASK_OPACITY", 0.7)

    # project_id = g.api.image.get_project_id(image_id)
    # project_meta = sly.ProjectMeta.from_json(g.JSON_METAS[project_id])
    project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(project_id))

    jann = g.api.annotation.download_json(image_id)
    ann = sly.Annotation.from_json(jann, project_meta)

    # for image, ann in zip([image], [ann]):
    rgba = u.get_rgba_np(
        ann, OUTPUT_WIDTH_PX, BBOX_THICKNESS_PERCENT, BBOX_OPACITY, FILLBBOX_OPACITY, MASK_OPACITY
    )

    arr = cv2.cvtColor(rgba.astype("uint8"), cv2.COLOR_RGBA2BGRA)

    success, im = cv2.imencode(".png", arr)
    # headers = {"Content-Disposition": 'inline; filename="test.png"'}
    headers = {"Cache-Control": "max-age=604800", "Content-Type": "image/png"}
    return Response(im.tobytes(), headers=headers, media_type="image/png")

import io

import cv2
import numpy as np
from fastapi import FastAPI
from starlette.responses import StreamingResponse

import src.globals as g
import src.utils as u
import supervisely as sly
from src.ui import settings_dict

app = FastAPI()


@app.post("/renders")
async def image_endpoint(project_id, image_id):
    OUTPUT_WIDTH_PX = settings_dict.get("OUTPUT_WIDTH_PX", 500)
    BBOX_THICKNESS_PERCENT = settings_dict.get("BBOX_THICKNESS_PERCENT", 0.005)
    BBOX_OPACITY = 1
    FILLBBOX_OPACITY = settings_dict.get("FILLBBOX_OPACITY", 0.2)
    MASK_OPACITY = settings_dict.get("MASK_OPACITY", 0.7)

    # project_id = g.api.image.get_project_id(image_id)
    project_meta = sly.ProjectMeta.from_json(g.JSON_METAS[project_id])

    jann = g.api.annotation.download_json(image_id)
    ann = sly.Annotation.from_json(jann, project_meta)

    # for image, ann in zip([image], [ann]):
    rgba = u.get_rgba_np(
        ann, OUTPUT_WIDTH_PX, BBOX_THICKNESS_PERCENT, BBOX_OPACITY, FILLBBOX_OPACITY, MASK_OPACITY
    )

    # res, im_png = cv2.imencode(".png", rgba)
    return StreamingResponse(io.BytesIO(rgba.tobytes()), media_type="image/png")

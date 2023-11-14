import cv2
import numpy as np
from fastapi import FastAPI
from starlette.responses import StreamingResponse

import src.globals as g
import supervisely as sly


def get_thickness(render: np.ndarray, thickness_percent: float) -> int:
    render_height, render_width, _ = render.shape
    return int(render_width * thickness_percent / 100)


def get_rgba_np(
    ann, OUTPUT_WIDTH_PX, BBOX_THICKNESS_PERCENT, BBOX_OPACITY, FILLBBOX_OPACITY, MASK_OPACITY
) -> np.ndarray:
    try:
        out_size = (int((ann.img_size[0] / ann.img_size[1]) * OUTPUT_WIDTH_PX), OUTPUT_WIDTH_PX)
        ann = ann.resize(out_size)
    
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
    
        alpha_mask = (MASK_OPACITY - np.all(render_mask == [0, 0, 0], axis=-1).astype("uint8")) * 255
        alpha_mask[alpha_mask < 0] = 0
    
        alpha_bbox = (BBOX_OPACITY - np.all(render_bbox == [0, 0, 0], axis=-1).astype("uint8")) * 255
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

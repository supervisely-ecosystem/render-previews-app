import json
import os
import shutil
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import requests
from tqdm import tqdm

import src.globals as g
import src.utils as u
import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    DatasetThumbnail,
    Editor,
    Empty,
    Image,
    Input,
    Progress,
    ProjectSelector,
    ProjectThumbnail,
    SelectDataset,
    SelectItem,
    SelectProject,
    SlyTqdm,
    Text,
)

progress_bar = Progress(show_percents=False)

input_tf_dest_dir = Input(placeholder="Please input here destination directory in Team files")

settings_dict = {
    "BBOX_THICKNESS_PERCENT": 0.5,
    "FILLBBOX_OPACITY": 0.2,
    "MASK_OPACITY": 0.7,
    "OUTPUT_WIDTH_PX": 500,
}
editor = Editor(initial_text=json.dumps(settings_dict, indent=4))
button_preview = Button(text="Preview Image")
button_save = Button(text="Save settings to team files")

# select1 = ProjectSelector()
# select_proj = SelectProject(workspace_/id=28, default_id=1195)

# proj_id = 1195
# select_ds = SelectDataset(default_id=1857, project_id=proj_id)

select_item = SelectItem(dataset_id=2343, compact=False)

# select_item.

img_orig, img_mask, img_overlap = Image(), Image(), Image()


card_1 = Card(
    title="Render settings",
    content=Container(
        widgets=[
            # select_ds,
            # input_tf_dest_dir,
            select_item,
            editor,
            Container([button_preview, button_save, Empty()], "horizontal", fractions=[1, 1, 8]),
            Container([img_orig, img_mask, img_overlap], "horizontal"),
        ]
    ),
)

progress_bar.hide()
button_save.disable()


@button_save.click
def save():
    img_mask.set("static/renders/1764964.png")


@button_preview.click
def preview():
    data_dict = json.loads(editor.get_text())

    item_id = select_item.get_selected_id()

    proj_id = g.api.image.get_project_id(select_item.get_selected_id())
    project = g.api.project.get_info_by_id(proj_id)
    project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(project.id))

    image = g.api.image.get_info_by_id(item_id)
    jann = g.api.annotation.download_json(item_id)
    ann = sly.Annotation.from_json(jann, project_meta)

    OUTPUT_WIDTH_PX = data_dict.get("OUTPUT_WIDTH_PX", 500)
    BBOX_THICKNESS_PERCENT = data_dict.get("BBOX_THICKNESS_PERCENT", 0.005)
    BBOX_OPACITY = data_dict.get("BBOX_OPACITY", 1)
    FILLBBOX_OPACITY = data_dict.get("FILLBBOX_OPACITY", 0)
    MASK_OPACITY = data_dict.get("MASK_OPACITY", 0.7)
    # u.save_preview(image, ann, dst_path)

    for image, ann in zip([image], [ann]):
        out_size = (int((image.height / image.width) * OUTPUT_WIDTH_PX), OUTPUT_WIDTH_PX)
        try:
            ann = ann.resize(out_size)
        except ValueError:
            continue

        render_mask = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
        render_bbox = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
        render_fillbbox = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)

        for label in ann.labels:
            label: sly.Label
            if type(label.geometry) == sly.Point:
                label.draw(render_mask, thickness=25)
            elif type(label.geometry) == sly.Rectangle:
                thickness = u.get_thickness(render_bbox, BBOX_THICKNESS_PERCENT)
                label.draw_contour(render_bbox, thickness=thickness)
                # label.draw(render_fillbbox)
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

        # alpha_fillbbox = (
        #     FILLBBOX_OPACITY - np.all(render_fillbbox == [0, 0, 0], axis=-1).astype("uint8")
        # ) * 255
        # alpha_fillbbox[alpha_fillbbox < 0] = 0

        # alpha = np.where(alpha_mask != 0, alpha_mask, alpha_fillbbox)
        alpha = np.where(alpha_bbox != 0, alpha_bbox, alpha_mask)

        rgba_mask = np.dstack((render_mask, alpha_mask))
        rgba_bbox = np.dstack((render_bbox, alpha_bbox))
        # render_fillbbox = np.dstack((render_fillbbox, alpha_fillbbox))

        # rgba = np.where(rgba_mask != 0, rgba_mask, render_fillbbox)
        rgba = np.where(rgba_bbox != 0, rgba_bbox, rgba_mask)

        # rgba = render_fillbbox

        orig = g.api.image.download_np(image.id)
        rgb = cv2.resize(orig, (out_size[1], out_size[0]))

        complement_alpha = 255 - alpha
        overlay_result = (complement_alpha[:, :, np.newaxis] / 255.0) * rgb + (
            alpha[:, :, np.newaxis] / 255.0
        ) * rgba[:, :, :3]
        rgb_overlap = np.clip(overlay_result, 0, 255).astype(np.uint8)

        local_path_rgb = os.path.join(os.getcwd(), "APP_DATA/resizedorigs", f"{image.id}.png")
        local_path_rgba = os.path.join(os.getcwd(), "APP_DATA/renders", f"{image.id}.png")
        local_path_overlap = os.path.join(os.getcwd(), "APP_DATA/overlaps", f"{image.id}.png")

        # if os.path.exists(local_path_rgba):
        #     os.remove(local_path_rgba)

        sly.image.write(local_path_rgb, rgb, remove_alpha_channel=True)
        sly.image.write(local_path_rgba, rgba, remove_alpha_channel=False)
        sly.image.write(local_path_overlap, rgb_overlap, remove_alpha_channel=True)

        img_orig.set(url=f"static/resizedorigs/{image.id}.png")
        img_mask.set(url=f"static/renders/{image.id}.png")
        img_overlap.set(url=f"static/overlaps/{image.id}.png")

    # janns = api.annotation.download_json_batch(dataset.id, [id for id in image_ids])
    # anns = [sly.Annotation.from_json(ann_json, project_meta) for ann_json in janns]

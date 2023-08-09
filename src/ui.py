import json
import os
import time

import cv2
import numpy as np

import src.globals as g
import src.utils as u
import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    Editor,
    Empty,
    Image,
    SelectItem,
    Text,
)

settings_dict = {
    "OUTPUT_WIDTH_PX": 500,
    "BBOX_THICKNESS_PERCENT": 0.5,
    "FILLBBOX_OPACITY": 0.2,
    "MASK_OPACITY": 0.7,
}
editor = Editor(initial_text=json.dumps(settings_dict, indent=4))

button_preview = Button(text="Preview Image")
button_save = Button(text="Save settings")
infotext = Text("Settings saved", "success")
select_item = SelectItem(dataset_id=2343, compact=False)

img_orig, img_mask, img_overlap = Image(), Image(), Image()


card_1 = Card(
    title="Render settings",
    content=Container(
        widgets=[
            select_item,
            editor,
            Container(
                [button_preview, button_save, infotext, Empty()],
                "horizontal",
                fractions=[1, 1, 1, 8],
            ),
            Container([img_orig, img_mask, img_overlap], "horizontal"),
        ]
    ),
)

infotext.hide()


def get_settings() -> dict:
    return settings_dict


@button_save.click
def save() -> None:
    global settings_dict
    settings_dict = json.loads(editor.get_text())
    infotext.show()
    time.sleep(2)
    infotext.hide()


@button_preview.click
def preview() -> None:
    settings = json.loads(editor.get_text())

    item_id = select_item.get_selected_id()

    proj_id = g.api.image.get_project_id(select_item.get_selected_id())

    # project = g.api.project.get_info_by_id(proj_id)
    project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(proj_id))
    # project_meta = g.JSON_METAS[proj_id]

    image = g.api.image.get_info_by_id(item_id)
    jann = g.api.annotation.download_json(item_id)
    ann = sly.Annotation.from_json(jann, project_meta)

    for image, ann in zip([image], [ann]):
        rgba, alpha, out_size = u.get_rgba_np(
            ann,
            settings.get("OUTPUT_WIDTH_PX", 500),
            settings.get("BBOX_THICKNESS_PERCENT", 0.5),
            settings.get("BBOX_OPACITY", 1),
            settings.get("FILLBBOX_OPACITY", 0.2),
            settings.get("MASK_OPACITY", 0.7),
        )

        orig = g.api.image.download_np(image.id)
        rgb = cv2.resize(orig, (out_size[1], out_size[0]))

        complement_alpha = 255 - alpha
        overlay_result = (complement_alpha[:, :, np.newaxis] / 255.0) * rgb + (
            alpha[:, :, np.newaxis] / 255.0
        ) * rgba[:, :, :3]
        rgb_overlap = np.clip(overlay_result, 0, 255).astype(np.uint8)

        local_path_rgb = os.path.join(g.STORAGE_DIR, "resizedorigs", f"{image.id}.png")
        local_path_rgba = os.path.join(g.STORAGE_DIR, "renders", f"{image.id}.png")
        local_path_overlap = os.path.join(g.STORAGE_DIR, "overlaps", f"{image.id}.png")

        sly.image.write(local_path_rgb, rgb, remove_alpha_channel=True)
        sly.image.write(local_path_rgba, rgba, remove_alpha_channel=False)
        sly.image.write(local_path_overlap, rgb_overlap, remove_alpha_channel=True)

        img_orig.set(url=f"static/resizedorigs/{image.id}.png")
        img_mask.set(url=f"static/renders/{image.id}.png")
        img_overlap.set(url=f"static/overlaps/{image.id}.png")

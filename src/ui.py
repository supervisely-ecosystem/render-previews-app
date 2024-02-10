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
    Grid,
    GridChart,
    GridPlot,
    Image,
    LineChart,
    LinePlot,
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
select_item = SelectItem(dataset_id=None, compact=False)

img_orig, img_mask, img_overlap = Image(), Image(), Image()

size1 = 10
x1 = list(range(size1))
y1 = np.random.randint(low=10, high=148, size=size1).tolist()
s1 = [{"x": x, "y": y} for x, y in zip(x1, y1)]

size2 = 30
x2 = list(range(size2))
y2 = np.random.randint(low=0, high=300, size=size2).tolist()
s2 = [{"x": x, "y": y} for x, y in zip(x2, y2)]

data_1 = {
    "title": "Line 1",
    "series": [{"name": "Line 1", "data": s1}],
}

data_2 = {
    "title": "Line 2",
    "series": [{"name": "Line 2", "data": s2}],
}

data_all = {
    "title": "All lines",
    "series": [{"name": "Max", "data": s1}, {"name": "Denis", "data": s2}],
}

line_chart = LineChart(
    title="Max vs Denis",
    series=[{"name": "loss", "data": []}],
    # series=[{"name": "Max", "data": []}, {"name": "Denis", "data": [s2]}],
    # xaxis_type="category",
)

butt = Button("add xy")
# line_chart.set_colors(
#     [
#         "rgb(0,0,0)",
#         "orange",
#     ]
# )
# "rgb(0,0,0)"

# grid_chart = Grid(widgets=[line_chart, line_chart, line_chart], columns=3, gap=0)


# LinePlot()
# LineChart()

# grid_plot = GridPlot(data=[data_1, data_2, data_all], columns=3)
# grid_chart = GridChart(data=[data_1, data_2, data_all], columns=3)+


data_max = {"title": "Max", "series": [{"name": "Max", "data": s1}]}
data_denis = {"title": "Denis", "series": [{"name": "Denis", "data": s2}]}

# grid_chart = GridChart(data=[data_max, data_denis], columns=2, gap=100)

# gs = GridChart(data="g")

# gs.add_scalars("g", {"s1": 2, "s2": 3}, 1)
# gs.add_scalar("g/s1", 3, 5)

data_max = {"title": "Max", "series": [{"name": "Max", "data": s1}]}
data_denis = {"title": "Denis", "series": [{"name": "Denis", "data": s2}]}

# grid_chart = GridChart(data=[data_max, data_denis], columns=2, gap=100)

grid_chart = GridChart(data=[data_max, data_denis], columns=2)

card_1 = Card(
    title="Grid Chart",
    content=Container(
        widgets=[
            # gs,
            grid_chart,
            # line_chart,
            butt,
            # grid_plot,
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


_x = 1000
_y = 1000


@butt.click
def tmp():
    global line_chart
    global _x, _y

    line_chart.add_to_series(line_chart.widget_id, [{"x": _x, "y": _y}])
    _y += 100
    _x += 100


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
            proj_id,
            image.id,
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

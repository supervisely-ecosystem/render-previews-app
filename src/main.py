import io
import zlib
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, Response
from PIL import Image
from tqdm import tqdm

import src.globals as g
import src.utils as u
import supervisely as sly
from src.ui import card_1, get_settings
from supervisely.app.widgets import Container

layout = Container(widgets=[card_1], direction="vertical")

static_dir = Path(g.STORAGE_DIR)
app = sly.Application(layout=layout, static_dir=static_dir)
server = app.get_server()

# 32796,
# data = g.api.annotation.download_batch(81589, [28437311, 28437307])[0]
# a = g.api.annotation.render_anns([28437311, 28437307])


# for idx, img_bytes in enumerate(a):
#     img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
#     img = Image.open(io.BytesIO(img_bytes))
#     img.save(f"image_{idx}.png")


item_id = 28437311
jann = g.api.annotation.download_json(item_id)
project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(32796))
ann = sly.Annotation.from_json(jann, project_meta)


render = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)

ann.draw(render, draw_class_names=False, draw_tags=True)


img = Image.fromarray(render)
img.save("image.png")


def dwnl_prj():
    project = g.api.project.get_info_by_id(34203)
    p = tqdm(
        desc="download",
        total=project.items_count,
    )
    # sly.download(g.api, project.id, "/tmp/vid/", progress_cb=p)
    sly.download(g.api, project.id, "/tmp/pclep/", progress_cb=p)
    print("3")


# def upl_prj_vid():
#     project_fs = sly.read_project("/tmp/vid/")
#     p = tqdm(
#         desc="upload",
#         total=project_fs.total_items,
#     )
#     sly.upload_video_project("/tmp/vid/", g.api, 691, progress_cb=p)
#     print("8")


# def upl_prj():
#     # project_fs = sly.read_project("/tmp/pics/")
#     # project_fs = sly.read_project("/tmp/vid/")
#     # project_fs = sly.read_project("/tmp/vol/")
#     project_fs = sly.read_project("/tmp/pclep/")
#     p = tqdm(
#         desc="upload",
#         total=project_fs.total_items,
#     )
#     # sly.upload("/tmp/pics/", g.api, 691, progress_cb=p)
#     # sly.upload("/tmp/vid/", g.api, 691)  # , progress_cb=p)
#     # sly.upload("/tmp/vol/", g.api, 691)  # , progress_cb=p)
#     sly.upload("/tmp/pclep/", g.api, 691, progress_cb=p)
#     print("4")

#     # shutil.rmtree("/tmp/lemons/")
#     # os.makedirs("/tmp/lemons/", exist_ok=True)


# def upl_prj_vid():
#     project_fs = sly.read_project("/tmp/vid/")
#     # p = tqdm(
#     #     desc="upload",
#     #     total=project_fs.total_items,
#     # )
#     sly.upload_video_project("/tmp/vid/", g.api, 691)  # , lo, progress_cb=p)
#     print("8")


# dwnl_prj()
# upl_prj()
# upl_prj_vid()


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

        jann = g.api.annotation.download_json(image_id)

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

    # import numpy as np

    # btmp = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
    # ann.draw(btmp)

    # g.api.annotation.render_ann(image_id, btmp, 300, 300, 1)
    return Response(im.tobytes(), headers=headers, media_type="image/png")

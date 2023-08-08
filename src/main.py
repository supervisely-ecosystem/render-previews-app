import multiprocessing
import os
from pathlib import Path

import uvicorn

import src.globals as g
import supervisely as sly
from src.endpoint import app as endpoint_app
from src.ui import card_1
from supervisely.app.widgets import Container

layout = Container(widgets=[card_1], direction="vertical")

static_dir = Path(g.STORAGE_DIR)
app = sly.Application(layout=layout, static_dir=static_dir)


def run_endpoint_app():
    uvicorn.run(endpoint_app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    endpoint_process = multiprocessing.Process(target=run_endpoint_app)

    endpoint_process.start()
    endpoint_process.join()

import os
from pathlib import Path

import src.globals as g
import supervisely as sly
from src.ui import card_1
from supervisely.app.widgets import Container

layout = Container(widgets=[card_1], direction="vertical")

static_dir = Path(g.STORAGE_DIR)
app = sly.Application(layout=layout, static_dir=static_dir)

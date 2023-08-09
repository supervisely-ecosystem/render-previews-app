import json
import os

from dotenv import load_dotenv

import supervisely as sly

if sly.is_development():
    load_dotenv("local.env")
    # load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv(os.path.expanduser("~/ninja.env"))

api: sly.Api = sly.Api.from_env()

TEAM_ID = sly.env.team_id()
WORKSPACE_ID = sly.env.workspace_id()
# PROJECT_ID = sly.env.project_id()
# DATASET_ID = sly.env.dataset_id(raise_not_found=False)

STORAGE_DIR = sly.app.get_data_dir()


def update_metas():
    sly.logger.info("Loading project metas. Please wait...")
    project_ids = [project.id for project in api.project.get_list(WORKSPACE_ID)]
    project_metas_json = [api.project.get_meta(id) for id in project_ids]
    sly.logger.info("Project meta successfully loaded")
    return {k: v for k, v in zip(project_ids, project_metas_json)}


JSON_METAS = update_metas()

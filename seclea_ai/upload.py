import os
import pickle  # nosec

# from sklearn.ensemble import HistGradientBoostingClassifier
from seclea_utils.models.sklearn.SkLearnModelManager import SkLearnModelManager
from seclea_utils.data.transmission.resquests_wrapper import RequestWrapper
from seclea_utils.auth.tokem_manager import update_token

USERNAME = 'onespanadmin'
PASSWORD = 'logmein1'

s = SkLearnModelManager()
trans_auth = RequestWrapper()


def login(username=USERNAME, password=PASSWORD, auth_url="https://auth.seclea.com",
          plat_url="https://platform.seclea.com"):
    s.manager.trans.server_root = "https://tristar-platform.seclea.com"
    trans_auth.server_root = "https://tristar-auth.seclea.com"
    credentials = {
        'username': username,
        'password': password
    }
    update_token(trans_plat=s.manager.trans, trans_auth=trans_auth, credentials=credentials)

    pass


def create_project(project_name: str, description: str):
    """

    :param project_name:
    :param description:
    :return:
    """
    s.manager.trans.url_path = "/collection/projects"
    r = s.manager.trans.send_json(
        {"name": project_name, "created_by": USERNAME, "description": description}
    )


def upload_dataset(dataset_path: str, project_name: str, dataset_id: str, metadata: dict):
    """

    :param dataset_path:
    :param project_name:
    :param dataset_id:
    :param metadata:
    :return:
    """
    s.manager.trans.url_path = "/collection/datasets"
    dataset_queryparams = {"project": project_name, "identifier": dataset_id, "metadata": metadata}
    s.manager.send_file(
        path=dataset_path, server_query_params=dataset_queryparams
    )


def save_training_run(model, project_name: str, dataset_id: str, training_run_id: str, metadata: dict, sequence_no=0):
    """

    :param project_name: "project-0"
    :param dataset_id: "test-dataset-0"
    :param training_run_id: "training-run-0"
    :param metadata:  {"type": "GradientBoostedClassifier"}
    :param sequence_no: 0
    :return:
    """
    s.manager.trans.url_path = "/collection/training-runs"
    s.manager.trans.send_json(
        {
            "project": project_name,
            "dataset": dataset_id,
            "identifier": training_run_id,
            "metadata": metadata,
        }
    )
    print("Saving model")

    save_path = s.save_model(model, f".seclea/{project_name}/{training_run_id}/model")

    s.manager.trans.url_path = f"/collection/training-runs/{training_run_id}/states"
    s.manager.send_file(
        save_path,
        {
            "sequence_num": str(sequence_no),
            "training_run": training_run_id,
        },
    )

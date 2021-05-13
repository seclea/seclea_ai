# from sklearn.ensemble import HistGradientBoostingClassifier
from seclea_utils.auth.token_manager import update_token
from seclea_utils.data.transmission.requests_wrapper import RequestWrapper
from seclea_utils.models.sklearn.SkLearnModelManager import SkLearnModelManager


class Seclea:
    def __init__(self, username, password, project_name=None):
        self.s = SkLearnModelManager()
        self.trans_auth = RequestWrapper()
        self.username = username
        self.password = password
        self.project_name = project_name

    def login(self, plat_url="https://platform.seclea.com", auth_url="https://auth.seclea.com"):
        self.s.manager.trans.server_root = plat_url
        self.trans_auth.server_root = auth_url
        credentials = {"username": self.username, "password": self.password}
        update_token(
            trans_plat=self.s.manager.trans, trans_auth=self.trans_auth, credentials=credentials
        )

    def create_project(self, description, project_name: str = None):
        """

        :param project_name:
        :param description:
        :return:
        """
        # check for overwriting project name
        if project_name is not None and self.project_name is not None:
            raise Exception("Project name is specified twice, this is probably a bug")
        # check for no project name at all
        if project_name is None and self.project_name is None:
            raise Exception("No project name specified, please provide one")
        # check for just using initialised project name.
        if project_name is None:
            project_name = self.project_name

        self.s.manager.trans.url_path = "/collection/projects"
        res = self.s.manager.trans.send_json(
            {"name": project_name, "created_by": self.username, "description": description}
        )
        # check for already created
        if not res.ok:
            raise Exception(f"Error creating project: {res.status_code} - {res.data}")
        else:
            self.project_name = project_name

    def upload_dataset(self, dataset_path: str, dataset_id: str, metadata: dict):
        """

        :param dataset_path:
        :param dataset_id:
        :param metadata:
        :return:
        """
        if self.project_name is None:
            raise Exception("You need to create a project before uploading a dataset")
        self.s.manager.trans.url_path = "/collection/datasets"
        dataset_queryparams = {
            "project": self.project_name,
            "identifier": dataset_id,
            "metadata": metadata,
        }
        res = self.s.manager.send_file(path=dataset_path, server_query_params=dataset_queryparams)
        if not res.ok:
            raise Exception(f"Error uploading dataset: {res.status_code} - {res.data}")

    def save_training_run(
        self, model, dataset_id: str, training_run_id: str, metadata: dict, sequence_no=0
    ):
        """

        :param model:
        :param dataset_id: "test-dataset-0"
        :param training_run_id: "training-run-0"
        :param metadata:  {"type": "GradientBoostedClassifier"}
        :param sequence_no: 0
        :return:
        """
        if self.project_name is None:
            raise Exception("You need to create a project before uploading a training run")
        self.s.manager.trans.url_path = "/collection/training-runs"
        self.s.manager.trans.send_json(
            {
                "project": self.project_name,
                "dataset": dataset_id,
                "identifier": training_run_id,
                "metadata": metadata,
            }
        )
        print("Saving model")

        save_path = self.s.save_model(model, f".seclea/{self.project_name}/{training_run_id}/model")

        self.s.manager.trans.url_path = f"/collection/training-runs/{training_run_id}/states"
        self.s.manager.send_file(
            save_path,
            {
                "sequence_num": str(sequence_no),
                "training_run": training_run_id,
            },
        )

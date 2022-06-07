import json

from seclea_ai.lib.seclea_utils.core.transmission import Transmission

# from ..errors import throws_api_err

root = "/collection/datasets"


# @throws_api_err
def post_dataset(
    transmission: Transmission,
    dataset_file_path: str,
    project_pk: str,
    organization_pk: str,
    name: str,
    metadata: dict,
    dataset_pk: str,
    parent_dataset_hash: str = None,
    delete=False,
):
    """
    @param transmission:
    @param dataset_file_path:
    @param project_pk:
    @param organization_pk:
    @param name:
    @param metadata:
    @param dataset_pk:
    @param parent_dataset_hash:
    @param delete: delete dataset file
    @return:
    """
    dataset_queryparams = {"project": project_pk, "organization": organization_pk}
    dataset_obj = {
        "project": (None, project_pk),
        "name": (None, name),
        "metadata": (None, json.dumps(metadata)),
        "hash": (None, str(dataset_pk)),
        "parent": (None, parent_dataset_hash),
        "dataset": ("dataset_name", open(dataset_file_path, "rb")),
    }
    res = transmission.send_file(
        url_path=f"{root}",
        obj=dataset_obj,
        query_params=dataset_queryparams,
    )
    return res

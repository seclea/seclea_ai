import json

from seclea_ai.seclea_utils.core.transmission import Transmission

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
    dataset_hash: str,
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
    @param dataset_hash:
    @param parent_dataset_hash:
    @param delete: delete dataset file
    @return:
    """
    dataset_queryparams = {
        "project": project_pk,
        "organization": organization_pk,
        "name": name,
        "metadata": json.dumps(metadata),
        "hash": str(dataset_hash),
        "parent": parent_dataset_hash,
    }
    res = transmission.send_file(
        url_path=f"{root}",
        file_path=dataset_file_path,
        query_params=dataset_queryparams,
        delete_file=delete,
    )
    return res

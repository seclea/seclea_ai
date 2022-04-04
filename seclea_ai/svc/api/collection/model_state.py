from seclea_ai.lib.seclea_utils.core.transmission import Transmission

root = "/collection/model-states"


def post_model_state(
    transmission: Transmission,
    model_state_file_path: str,
    organization_pk: str,
    project_pk: str,
    training_run_pk: str,
    sequence_num: int,
    final_state,
    delete=False,
):
    """

    @param transmission:
    @param model_state_file_path:
    @param organization_pk:
    @param project_pk:
    @param training_run_pk:
    @param sequence_num:
    @param final_state:
    @param delete:
    @return:
    """
    res = transmission.send_file(
        url_path="/collection/model-states",
        file_path=model_state_file_path,
        query_params={
            "organization": organization_pk,
            "project": project_pk,
            "sequence_num": sequence_num,
            "training_run": training_run_pk,
            "final_state": final_state,
        },
        delete_file=delete,
    )
    return res

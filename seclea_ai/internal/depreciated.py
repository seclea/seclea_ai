import copy

def get_dataset_type(dataset: DataFrame) -> str:
    if not np.issubdtype(dataset.index.dtype, np.integer):
        try:
            pd.to_datetime(dataset.index.values)
        except (ParserError, ValueError):  # Can't convert some
            return "tabular"
        return "time_series"
    return "tabular"

def upload_training_run(
        self,
        model: Tracked,
        train_dataset: DataFrame,
        test_dataset: DataFrame = None,
        val_dataset: DataFrame = None,
) -> None:
    """
    Takes a model and extracts the necessary data for uploading the training run.

    :param model: An ML Model instance. This should be one of {sklearn.Estimator, xgboost.Booster, lgbm.Boster}.

    :param train_dataset: DataFrame The Dataset that the model is trained on.

    :param test_dataset: DataFrame The Dataset that the model is trained on.

    :param val_dataset: DataFrame The Dataset that the model is trained on.

    :return: None

    Example::

        >>> seclea = SecleaAI(project_name="Test Project")
        >>> dataset = pd.read_csv(<dataset_name>)
        >>> model = LogisticRegressionClassifier()
        >>> model.fit(x, y)
        >>> seclea.upload_training_run(
                model,
                framework=seclea_ai.Frameworks.SKLEARN,
                dataset_name="Test Dataset",
            )
    """
    # TODO check if we need this auth check here or if we can do better.
    self._api.authenticate()

    # validate the splits? maybe later when we have proper Dataset class to manage these things.
    dataset_ids = [
        dataset_hash(dataset, self._project_id)
        for dataset in [train_dataset, test_dataset, val_dataset]
        if dataset is not None
    ]

    # Model stuff
    model_name = model.__class__.__name__
    # check the model exists upload if not TODO convert to add to queue
    model_type_id = self._set_model(
        model_name=model_name, framework=model.object_manager.framework
    )

    # check the latest training run TODO extract all this stuff
    training_runs_res = self._api.get_training_runs(
        project_id=self._project_id,
        organization_id=self._organization,
        model=model_type_id,
    )
    training_runs = training_runs_res.json()

    # Create the training run name
    largest = -1
    for training_run in training_runs:
        num = int(training_run["name"].split(" ")[2])
        if num > largest:
            largest = num
    training_run_name = f"Training Run {largest + 1}"

    # extract params from the model
    params = model.object_manager.get_params(model)

    # search for datasets in local db? Maybe not needed..

    # create record
    self._db.connect()
    training_run_record = Record.create(
        entity="training_run", status=RecordStatus.IN_MEMORY.value
    )

    # sent training run for upload.
    training_run_details = {
        "entity": "training_run",
        "record_id": training_run_record.id,
        "project": self._project_id,
        "training_run_name": training_run_name,
        "model_id": model_type_id,
        "dataset_ids": dataset_ids,
        "params": params,
    }
    self._director.send_entity(training_run_details)

    # create local db record
    model_state_record = Record.create(
        entity="model_state",
        status=RecordStatus.IN_MEMORY.value,
        dependencies=[training_run_record.id],
    )
    self._db.close()

    # send model state for save and upload
    # TODO make a function interface rather than the queue interface. Need a response to confirm it is okay.
    model_state_details = {
        "entity": "model_state",
        "record_id": model_state_record.id,
        "model": model,
        "sequence_num": 0,
        "final": True,
        "model_manager": model.object_manager.framework,  # TODO move framework to sender.
    }
    self._director.store_entity(model_state_details)
    self._director.send_entity(model_state_details)

def _set_model(self, model_name: str, framework: str) -> int:
    """
    Set the model for this session.
    Checks if it has already been uploaded. If not it will upload it.

    :param model_name: The name for the architecture/algorithm. eg. "GradientBoostedMachine" or "3-layer CNN".

    :return: int The model id.

    :raises: ValueError - if the framework is not one of the supported frameworks or if there is an issue uploading
     the model.
    """
    res = self._api.get_models(
        organization_id=self._organization,
        project_id=self._project_id,
        name=model_name,
        framework=framework,
    )
    models = res.json()
    # not checking for more because there is a unique constraint across name and framework on backend.
    if len(models) == 1:
        return models[0]["id"]
    # if we got here that means that the model has not been uploaded yet. So we upload it.
    res = self._api.upload_model(
        organization_id=self._organization,
        project_id=self._project_id,
        model_name=model_name,
        framework_name=framework,
    )
    # TODO find out if this checking is ever needed - ie does it ever not return the created model object?
    try:
        model_id = res.json()["id"]
    except KeyError:
        traceback.print_exc()
        resp = self._api.get_models(
            organization_id=self._organization,
            project_id=self._project_id,
            name=model_name,
            framework=framework,
        )
        model_id = resp.json()[0]["id"]
    return model_id


def upload_dataset(
        self,
        dataset: Tracked,
        dataset_name: str,
        metadata: Dict,
        transformations: List[DatasetTransformation] = None,
) -> None:
    """
    Uploads a dataset.

    :param dataset: DataFrame, Path or list of paths to the dataset.
        If a list then they must be split by row only and all
        files must contain column names as a header line.

    :param dataset_name: The name of the dataset.

    :param metadata: Any metadata about the dataset. Note that if using a Path or list of Paths then if there is an
        index that you use when loading the data, it must be specified in the metadata.

    :param transformations: A list of DatasetTransformation's.

                    If your Dataset is large try call this function more often with less DatasetTransformations
                    as the function currently requires no. DatasetTransformations x Dataset size memory to function.

                    See DatasetTransformation for more details.

    :return: None

    Example:: TODO update docs
        >>> seclea = SecleaAI(project_name="Test Project")
        >>> dataset = pd.read_csv("/test_folder/dataset_file.csv")
        >>> dataset_metadata = {"index": "TransactionID", "outcome_name": "isFraud", "continuous_features": ["TransactionDT", "TransactionAmt"]}
        >>> seclea.upload_dataset(dataset=dataset, dataset_name="Multifile Dataset", metadata=dataset_metadata)

    Example with file::

        >>> seclea.upload_dataset(dataset="/test_folder/dataset_file.csv", dataset_name="Test Dataset", metadata={})
        >>> seclea = SecleaAI(project_name="Test Project", organization="Test Organization")

    Assuming the files are all in the /test_folder/dataset directory.
    Example with multiple files::

        >>> files = os.listdir("/test_folder/dataset")
        >>> seclea = SecleaAI(project_name="Test Project")
        >>> dataset_metadata = {"index": "TransactionID", "outcome_name": "isFraud", "continuous_features": ["TransactionDT", "TransactionAmt"]}
        >>> seclea.upload_dataset(dataset=files, dataset_name="Multifile Dataset", metadata=dataset_metadata)


    """
    # processing the final dataset - make sure it's a DataFrame

    # TODO replace with dataset_hash fn
    dataset_id = dataset, self._project_id

    if transformations is not None:
        parent = Tracked(self._assemble_dataset(*transformations[0].raw_data_kwargs.values()))

        #####
        # Validate parent exists and get metadata - check how often on portal, maybe remove?
        #####
        parent_dset_id = parent.object_manager.hash(parent, self._project_id)
        # check parent exists - check local db if not else error.
        try:
            res = self._api.get_dataset(
                dataset_id=str(parent_dset_id),
                organization_id=self._organization,
                project_id=self._project_id,
            )
        except NotFoundError:
            # check local db
            self._db.connect()
            parent_record = Record.get_or_none(Record.key == parent_dset_id)
            self._db.close()
            if parent_record is not None:
                parent_metadata = parent_record.dataset_metadata
            else:
                raise AssertionError(
                    "Parent Dataset does not exist on the Platform or locally. Please check your arguments and "
                    "that you have uploaded the parent dataset already"
                )
        else:
            parent_metadata = res.json()["metadata"]
        #####

        upload_queue = self._generate_intermediate_datasets(
            transformations=transformations,
            dataset_name=dataset_name,
            dataset_id=dataset.object_manager.hash(dataset, self._project_id),
            user_metadata=metadata,
            parent=parent,
            parent_metadata=parent_metadata,
        )

        # upload all the datasets and transformations.
        for up_kwargs in upload_queue:
            up_kwargs["project"] = self._project_id
            # add to storage and sending queues
            if up_kwargs["entity"] == "dataset":
                self._director.store_entity(up_kwargs)
            self._director.send_entity(up_kwargs)
        return

    # this only happens if this has no transformations ie. it is a Raw Dataset.

    # validation
    features = list(getattr(dataset, 'columns', []))
    categorical_features = list(set(features) - set(metadata.get("continuous_features", [])))
    categorical_values = [{col: dataset[col].unique().tolist()} for col in categorical_features]
    metadata_defaults_spec = dict(
        continuous_features=[],
        outcome_name=None,
        num_samples=len(dataset),
        favourable_outcome=None,
        unfavourable_outcome=None,
        dataset_type=self._get_dataset_type(dataset),
        index=0 if dataset.index.name is None else dataset.index.name,
        split=None,
        features=features,
        categorical_features=categorical_features,
        categorical_values=categorical_values
    )
    metadata = {**metadata_defaults_spec, **metadata}

    # create local db record.
    # TODO make lack of parent more obvious??
    self._db.connect()
    dataset_record = Record.create(
        entity="dataset",
        status=RecordStatus.IN_MEMORY.value,
        key=dataset.object_manager.hash(dataset, self._project_id),
        dataset_metadata=metadata,
    )
    self._db.close()

    # New arch
    dataset_upload_kwargs = {
        "entity": "dataset",
        "record_id": dataset_record.id,
        "dataset": dataset,
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "metadata": metadata,
        "project": self._project_id,
    }
    # add to storage and sending queues
    self._director.store_entity(dataset_upload_kwargs)
    self._director.send_entity(dataset_upload_kwargs)


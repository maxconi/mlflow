"""
The ``mlflow.sklearn`` module provides an API for logging and loading scikit-learn models. This
module exports scikit-learn models with the following flavors:

Python (native) `pickle <https://scikit-learn.org/stable/modules/model_persistence.html>`_ format
    This is the main flavor that can be loaded back into scikit-learn.

:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

import os
import pickle
import yaml
import gorilla
import tempfile

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, INTERNAL_ERROR
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.autologging_utils import try_mlflow_log, log_fn_args_as_params

FLAVOR_NAME = "sklearn"

SERIALIZATION_FORMAT_PICKLE = "pickle"
SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"

SUPPORTED_SERIALIZATION_FORMATS = [
    SERIALIZATION_FORMAT_PICKLE,
    SERIALIZATION_FORMAT_CLOUDPICKLE
]

def get_default_conda_env(include_cloudpickle=False):
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import sklearn
    pip_deps = None
    if include_cloudpickle:
        import cloudpickle
        pip_deps = ["cloudpickle=={}".format(cloudpickle.__version__)]
    return _mlflow_conda_env(
        additional_conda_deps=[
            "scikit-learn={}".format(sklearn.__version__),
        ],
        additional_pip_deps=pip_deps,
        additional_conda_channels=None
    )


def save_model(sk_model, path, conda_env=None, mlflow_model=Model(),
               serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE):
    """
    Save a scikit-learn model to a path on the local file system.

    :param sk_model: scikit-learn model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If `None`, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'scikit-learn=0.19.2'
                            ]
                        }

    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the formats listed in
                                 ``mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS``. The Cloudpickle
                                 format, ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``,
                                 provides better cross-system compatibility by identifying and
                                 packaging code dependencies with the serialized model.

    >>> import mlflow.sklearn
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree
    >>> iris = load_iris()
    >>> sk_model = tree.DecisionTreeClassifier()
    >>> sk_model = sk_model.fit(iris.data, iris.target)
    >>> #Save the model in cloudpickle format
    >>> #set path to location for persistence
    >>> sk_path_dir_1 = ...
    >>> mlflow.sklearn.save_model(
    >>>         sk_model, ion_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
    >>>sk_path_dir_1,
    >>>         serializat
    >>> #Save the model in pickle format
    >>> #set path to location for persistence
    >>> sk_path_dir_2 = ...
    >>> mlflow.sklearn.save_model(sk_model, sk_path_dir_2,
    >>>                           serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
    """
    import sklearn
    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
                message=(
                    "Unrecognized serialization format: {serialization_format}. Please specify one"
                    " of the following supported formats: {supported_formats}.".format(
                        serialization_format=serialization_format,
                        supported_formats=SUPPORTED_SERIALIZATION_FORMATS)),
                error_code=INVALID_PARAMETER_VALUE)

    if os.path.exists(path):
        raise MlflowException(message="Path '{}' already exists".format(path),
                              error_code=RESOURCE_ALREADY_EXISTS)
    os.makedirs(path)
    model_data_subpath = "model.pkl"
    _save_model(sk_model=sk_model, output_path=os.path.join(path, model_data_subpath),
                serialization_format=serialization_format)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env(
            include_cloudpickle=serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE)
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn", data=model_data_subpath,
                        env=conda_env_subpath)
    mlflow_model.add_flavor(FLAVOR_NAME,
                            pickled_model=model_data_subpath,
                            sklearn_version=sklearn.__version__,
                            serialization_format=serialization_format)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(sk_model, artifact_path, conda_env=None,
              serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE, registered_model_name=None):
    """
    Log a scikit-learn model as an MLflow artifact for the current run.

    :param sk_model: scikit-learn model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If `None`, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'scikit-learn=0.19.2'
                            ]
                        }

    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the formats listed in
                                 ``mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS``. The Cloudpickle
                                 format, ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``,
                                 provides better cross-system compatibility by identifying and
                                 packaging code dependencies with the serialized model.
    :param registered_model_name: Note:: Experimental: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.

    >>> import mlflow
    >>> import mlflow.sklearn
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree
    >>> iris = load_iris()
    >>> sk_model = tree.DecisionTreeClassifier()
    >>> sk_model = sk_model.fit(iris.data, iris.target)
    >>> #set the artifact_path to location where experiment artifacts will be saved
    >>> #log model params
    >>> mlflow.log_param("criterion", sk_model.criterion)
    >>> mlflow.log_param("splitter", sk_model.splitter)
    >>> #log model
    >>> mlflow.sklearn.log_model(sk_model, "sk_models")
    """
    return Model.log(artifact_path=artifact_path,
                     flavor=mlflow.sklearn,
                     sk_model=sk_model,
                     conda_env=conda_env,
                     serialization_format=serialization_format,
                     registered_model_name=registered_model_name)


def _load_model_from_local_file(path):
    """Load a scikit-learn model saved as an MLflow artifact on the local file system."""
    # TODO: we could validate the scikit-learn version here
    with open(path, "rb") as f:
        # Models serialized with Cloudpickle can be deserialized using Pickle; in fact,
        # Cloudpickle.load() is just a redefinition of pickle.load(). Therefore, we do
        # not need to check the serialization format of the model before deserializing.
        return pickle.load(f)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``sklearn`` flavor.
    """
    return _load_model_from_local_file(path)


def _save_model(sk_model, output_path, serialization_format):
    """
    :param sk_model: The scikit-learn model to serialize.
    :param output_path: The file path to which to write the serialized model.
    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`` or
                                 ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    """
    with open(output_path, "wb") as out:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            pickle.dump(sk_model, out)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle
            cloudpickle.dump(sk_model, out)
        else:
            raise MlflowException(
                    message="Unrecognized serialization format: {serialization_format}".format(
                        serialization_format=serialization_format),
                    error_code=INTERNAL_ERROR)


def load_model(model_uri):
    """
    Load a scikit-learn model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model, for example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.

    :return: A scikit-learn model.

    >>> import mlflow.sklearn
    >>> sk_model = mlflow.sklearn.load_model("runs:/96771d893a5e46159d9f3b49bf9013e2/"sk_models")
    >>> #use Pandas DataFrame to make predictions
    >>> pandas_df = ...
    >>> predictions = sk_model.predict(pandas_df)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    sklearn_model_artifacts_path = os.path.join(local_model_path, flavor_conf['pickled_model'])
    return _load_model_from_local_file(path=sklearn_model_artifacts_path)

@experimental
def autolog():
    """
    Enable automatic logging from Keras to MLflow.
    Logs loss and any other metrics specified in the fit
    function, and optimizer data as parameters. Model checkpoints
    are logged as artifacts to a 'models' directory.

    EarlyStopping Integration with Keras Automatic Logging

    MLflow will detect if an ``EarlyStopping`` callback is used in a ``fit()``/``fit_generator()``
    call, and if the ``restore_best_weights`` parameter is set to be ``True``, then MLflow will
    log the metrics associated with the restored model as a final, extra step. The epoch of the
    restored model will also be logged as the metric ``restored_epoch``.
    This allows for easy comparison between the actual metrics of the restored model and
    the metrics of other models.

    If ``restore_best_weights`` is set to be ``False``,
    then MLflow will not log an additional step.

    Regardless of ``restore_best_weights``, MLflow will also log ``stopped_epoch``,
    which indicates the epoch at which training stopped due to early stopping.

    If training does not end due to early stopping, then ``stopped_epoch`` will be logged as ``0``.

    MLflow will also log the parameters of the EarlyStopping callback,
    excluding ``mode`` and ``verbose``.
    """
    import keras

    class __MLflowKerasCallback(keras.callbacks.Callback):
        """
        Callback for auto-logging metrics and parameters.
        Records available logs after each epoch.
        Records model structural information as params when training begins
        """
        def on_train_begin(self, logs=None):  # pylint: disable=unused-argument
            try_mlflow_log(mlflow.log_param, 'num_layers', len(self.model.layers))
            try_mlflow_log(mlflow.log_param, 'optimizer_name', type(self.model.optimizer).__name__)
            if hasattr(self.model.optimizer, 'lr'):
                lr = self.model.optimizer.lr if \
                    type(self.model.optimizer.lr) is float \
                    else keras.backend.eval(self.model.optimizer.lr)
                try_mlflow_log(mlflow.log_param, 'learning_rate', lr)
            if hasattr(self.model.optimizer, 'epsilon'):
                epsilon = self.model.optimizer.epsilon if \
                    type(self.model.optimizer.epsilon) is float \
                    else keras.backend.eval(self.model.optimizer.epsilon)
                try_mlflow_log(mlflow.log_param, 'epsilon', epsilon)

            sum_list = []
            self.model.summary(print_fn=sum_list.append)
            summary = '\n'.join(sum_list)
            try_mlflow_log(mlflow.set_tag, 'model_summary', summary)

            tempdir = tempfile.mkdtemp()
            try:
                summary_file = os.path.join(tempdir, "model_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write(summary)
                try_mlflow_log(mlflow.log_artifact, local_path=summary_file)
            finally:
                shutil.rmtree(tempdir)

        def on_epoch_end(self, epoch, logs=None):
            if not logs:
                return
            try_mlflow_log(mlflow.log_metrics, logs, step=epoch)

        def on_train_end(self, logs=None):
            try_mlflow_log(log_model, self.model, artifact_path='model')

    def _early_stop_check(callbacks):
        if LooseVersion(keras.__version__) < LooseVersion('2.3.0'):
            es_callback = keras.callbacks.EarlyStopping
        else:
            es_callback = keras.callbacks.callbacks.EarlyStopping
        for callback in callbacks:
            if isinstance(callback, es_callback):
                return callback
        return None

    def _log_early_stop_callback_params(callback):
        if callback:
            try:
                earlystopping_params = {'monitor': callback.monitor,
                                        'min_delta': callback.min_delta,
                                        'patience': callback.patience,
                                        'baseline': callback.baseline,
                                        'restore_best_weights': callback.restore_best_weights}
                try_mlflow_log(mlflow.log_params, earlystopping_params)
            except Exception:  # pylint: disable=W0703
                return

    def _get_early_stop_callback_attrs(callback):
        try:
            return callback.stopped_epoch, callback.restore_best_weights, callback.patience
        except Exception:  # pylint: disable=W0703
            return None

    def _log_early_stop_callback_metrics(callback, history):
        if callback:
            callback_attrs = _get_early_stop_callback_attrs(callback)
            if callback_attrs is None:
                return
            stopped_epoch, restore_best_weights, patience = callback_attrs
            try_mlflow_log(mlflow.log_metric, 'stopped_epoch', stopped_epoch)
            # Weights are restored only if early stopping occurs
            if stopped_epoch != 0 and restore_best_weights:
                restored_epoch = stopped_epoch - max(1, patience)
                try_mlflow_log(mlflow.log_metric, 'restored_epoch', restored_epoch)
                restored_metrics = {key: history.history[key][restored_epoch]
                                    for key in history.history.keys()}
                # Checking that a metric history exists
                metric_key = next(iter(history.history), None)
                if metric_key is not None:
                    last_epoch = len(history.history[metric_key])
                    try_mlflow_log(mlflow.log_metrics, restored_metrics, step=last_epoch)

    def _run_and_log_function(self, original, args, kwargs, unlogged_params, callback_arg_index):
        if not mlflow.active_run():
            try_mlflow_log(mlflow.start_run)
            auto_end_run = True
        else:
            auto_end_run = False

        log_fn_args_as_params(original, args, kwargs, unlogged_params)
        early_stop_callback = None

        # Checking if the 'callback' argument of the function is set
        if len(args) > callback_arg_index:
            tmp_list = list(args)
            early_stop_callback = _early_stop_check(tmp_list[callback_arg_index])
            tmp_list[callback_arg_index] += [__MLflowKerasCallback()]
            args = tuple(tmp_list)
        elif 'callbacks' in kwargs:
            early_stop_callback = _early_stop_check(kwargs['callbacks'])
            kwargs['callbacks'] += [__MLflowKerasCallback()]
        else:
            kwargs['callbacks'] = [__MLflowKerasCallback()]

        _log_early_stop_callback_params(early_stop_callback)

        history = original(self, *args, **kwargs)

        _log_early_stop_callback_metrics(early_stop_callback, history)

        if auto_end_run:
            try_mlflow_log(mlflow.end_run)

        return history

    @gorilla.patch(keras.Model)
    def fit(self, *args, **kwargs):
        original = gorilla.get_original_attribute(keras.Model, 'fit')
        unlogged_params = ['self', 'x', 'y', 'callbacks', 'validation_data', 'verbose']
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 5)

    @gorilla.patch(keras.Model)
    def fit_generator(self, *args, **kwargs):
        original = gorilla.get_original_attribute(keras.Model, 'fit_generator')
        unlogged_params = ['self', 'generator', 'callbacks', 'validation_data', 'verbose']
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 4)

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    gorilla.apply(gorilla.Patch(keras.Model, 'fit', fit, settings=settings))
    gorilla.apply(gorilla.Patch(keras.Model, 'fit_generator', fit_generator, settings=settings))


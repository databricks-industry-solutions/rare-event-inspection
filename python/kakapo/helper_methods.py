from hyperopt import fmin, tpe, hp, Trials, SparkTrials, STATUS_OK
from hyperopt.pyll.base import scope
from pyod.models.iforest import IForest
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.abod import ABOD
from pyod.models.inne import INNE
from emmv import emmv_scores
import mlflow

from .pyodwrapper import *


def get_default_model_space():
    """
  Generate a default model space for pyodflow
  """
    model_space = {
        "ecod": ECOD,
        "abod": ABOD,
        "iforest": IForest,
        "copod": COPOD,
        "inne": INNE
    }
    return model_space


def enrich_default_model_space(model_dict):
    """
  Enrich the default model space for pyodflow
  """
    return dict(get_default_model_space(), **model_dict)


def get_default_search_space():
    """
  Generate a default hyperopt search space
  """
    space_list = [
        {
            'type': 'iforest',
            'n_estimators': scope.int(hp.quniform('iforest.n_estimators', 100, 500, 25)),
            'max_features': hp.quniform('iforest.max_features', 0.5, 1, 0.1)
        },
        {
            'type': 'inne',
            'n_estimators': scope.int(hp.quniform('inne.n_estimators', 100, 500, 25)),
            'max_samples': hp.quniform('inne.max_samples', 0.1, 1, 0.1)
        },
        {
            'type': 'abod',
            'n_neighbors': scope.int(hp.quniform('abod.n_neighbors', 5, 20, 5))
        },
        {
            'type': 'ecod'
        },
        {
            'type': 'copod'
        },
    ]
    return space_list


def enrich_default_search_space(space_list):
    """
  Enrich the default hyperopt search space
  """
    return get_default_search_space() + space_list


def train_outlier_detection(params, model_space, X_train, X_test, y_test, ground_truth_flag):
    """
    Train an outlier detection model using the pyfunc wrapper for PyOD algorithms and log into mlflow
    """
    mlflow.autolog(disable=False)
    mlflow.set_tag("model_type", params["type"])

    model = PyodWrapper(**params)
    model.set_model_space(model_space)
    model.fit(X_train)

    y_test_pred = model.predict(None, X_test)

    # Get model input and output signatures
    model_input_df = X_train
    model_output_df = y_test_pred
    model_signature = infer_signature(model_input_df, model_output_df)

    # log our model to mlflow
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model,
        signature=model_signature
    )

    if (ground_truth_flag):
        score = roc_auc_score(y_score=y_test_pred, y_true=y_test)
    else:
        score = emmv_scores(model, X_test)["em"]

    return {'loss': -score, 'status': STATUS_OK}

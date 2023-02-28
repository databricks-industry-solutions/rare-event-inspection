# Databricks notebook source
# MAGIC %pip install pyod emmv Jinja2==3.0.3

# COMMAND ----------

import sys
import uuid

from pyod.utils.data import generate_data
from pyspark.sql.functions import struct, col, when, array
from pyspark.sql.types import DoubleType

# COMMAND ----------

code_location = "Repos/iliya.kostov@databricks.com/kakapo/kakapo"
sys.path.append(f"/Workspace/{code_location}")
from kakapo import *

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Generate synthetic data
# MAGIC Using pyod's inbuilt generate_data method to create a simple synthetic data set with 5 features and a certain
# MAGIC proportion of "outliers" and split the data into train and test sets

# COMMAND ----------

contamination = 0.1  # percentage of outliers
n_train = 2000  # number of training points
n_test = 500  # number of testing points

# Generate sample data
X_train, X_test, y_train, y_test = \
    generate_data(n_train=n_train,
                  n_test=n_test,
                  n_features=5,
                  contamination=contamination,
                  random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2. Using pyodflow with the default settings
# MAGIC
# MAGIC Train multiple outlier detection models and perform hyperparameter search in parallel using pyodflow's default
# MAGIC settings for model space and hyperparameter search space

# COMMAND ----------

# Load default model space
model_space = get_default_model_space()
print("Default model space: {}".format(model_space))

# COMMAND ----------

# Load default hyper param search space
search_space = get_default_search_space()
print("Default search space: {}".format(search_space))

# COMMAND ----------

# Load search space into hyperopt
space = hp.choice('model_type', search_space)

# COMMAND ----------

# Controls whether or not we have ground truth labels to evaluate the outlier models
GROUND_TRUTH_OD_EXISTS = True

# Unique run ID when saving MLFlow experiment
uid = uuid.uuid4().hex

# COMMAND ----------

with mlflow.start_run(run_name=uid):
    best_params = fmin(
        trials=SparkTrials(parallelism=10),
        fn=lambda params: train_outlier_detection(params, model_space, X_train, X_test, y_test, GROUND_TRUTH_OD_EXISTS),
        space=space,
        algo=tpe.suggest,
        max_evals=50
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 3. Extending pyodflow's default settings
# MAGIC
# MAGIC Extend the default settings of pyodflow and include a new model in the model space and a selection of hyperparameter options to the search space

# COMMAND ----------

# Import a new model we want to include in our space
from pyod.models.cof import COF

# COMMAND ----------

# Enrich the default model space
model_space = enrich_default_model_space({"cof": COF})
print("Enriched model space: {}".format(model_space))

# COMMAND ----------

additional_params = [
    {
        'type': 'cof',
        'n_neighbors': scope.int(hp.quniform('cof.n_neighbors', 5, 20, 5))
    }
]

# Enrich the default hyper param search space
search_space = enrich_default_search_space(additional_params)
print("Enriched search space: {}".format(search_space))

# COMMAND ----------

# Load enriched search space into hyperopt
space = hp.choice('model_type', search_space)

# COMMAND ----------

# Controls whether or not we have ground truth labels to evaluate the outlier models
GROUND_TRUTH_OD_EXISTS = False

# Unique run ID when saving MLFlow experiment
uid = uuid.uuid4().hex

# COMMAND ----------

with mlflow.start_run(run_name=uid):
    best_params = fmin(
        trials=SparkTrials(parallelism=10),
        fn=lambda params: train_outlier_detection(params, model_space, X_train, X_test, y_test, GROUND_TRUTH_OD_EXISTS),
        space=space,
        algo=tpe.suggest,
        max_evals=50
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4. Using trained models for inference
# MAGIC
# MAGIC Load a trained model and carry out predictions on a dataset

# COMMAND ----------

mlflow.search_runs()

# COMMAND ----------

metric = "loss"
parentRunId = "'<PARENT-RUN-ID>'"

# Get all child runs on current experiment
runs = mlflow.search_runs(filter_string=f"tags.mlflow.parentRunId = {parentRunId}", order_by=[f"metrics.{metric} ASC"])
runs = runs.where(runs['status'] == 'FINISHED')

# Get best run id
best_run_id = runs.loc[0, 'run_id']

print(best_run_id)

# COMMAND ----------

sdf_X_test = spark.createDataFrame(X_test.tolist(), schema=["col1", "col2", "col3", "col4", "col5"])
sdf_X_test.display()

# COMMAND ----------

logged_model = f'runs:/{best_run_id}/model'

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')

# Predict on a Spark DataFrame.
sdf_X_test = sdf_X_test.withColumn('predictions', loaded_model(struct(*map(col, sdf_X_test.columns))))

# COMMAND ----------

sdf_X_test.display()

# COMMAND ----------

# def get_best_run(experiment_name, metric, uid):
#   exp = mlflow.get_experiment_by_name(experiment_name)
#   df = mlflow.search_runs([exp.experiment_id], order_by=[f"metrics.{metric} DESC"])
#   df = df.where(df['status'] == 'FINISHED').where(df['tags.uuid'] == uid)
#   best_run_id = df.loc[0,'run_id']
#   return best_run_id

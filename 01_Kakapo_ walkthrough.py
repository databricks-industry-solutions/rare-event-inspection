# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/rare-event-inspection

# COMMAND ----------

# MAGIC %pip install databricks-kakapo pyod --quiet

# COMMAND ----------

from hyperopt import fmin, tpe, hp, Trials, SparkTrials, STATUS_OK
from hyperopt.pyll.base import scope
from pyspark.sql.functions import struct, col, when, array
from pyspark.sql.types import DoubleType
from pyod.utils.data import generate_data

import mlflow
import kakapo
import uuid
import sys

# COMMAND ----------

# MAGIC %md
# MAGIC Create a mlflow experiment to track the results of the hyperparameter search.

# COMMAND ----------

user = spark.sql("select current_user()").take(1)[0][0]
mlflow.set_experiment(f"/Users/{user}/kakapo")


# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1. Generate synthetic data
# MAGIC Use pyod's inbuilt generate_data method to create a simple synthetic data set with 5 features and a certain proportion of "outliers" and split the data into train and test sets

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
# MAGIC ### 2. Define the model universe and hyper parameter search space using kakapo's default settings
# MAGIC 
# MAGIC Train multiple outlier detection models and perform hyper parameter search in parallel using kakapo's default settings for model space and hyper paramemeter search space

# COMMAND ----------

# Load default model space
model_space = kakapo.get_default_model_space()
print("Default model space: {}".format(model_space))

# COMMAND ----------

# Load default hyper param search space
search_space = kakapo.get_default_search_space()
print("Default search space: {}".format(search_space))

# COMMAND ----------

# Load search space into hyperopt
space = hp.choice('model_type', search_space)

# COMMAND ----------

# Controls wether or not we have ground truth labels to evaluate the outlier models
GROUND_TRUTH_OD_EXISTS = True

# Unique run ID when saving MLFlow experiment
uid = uuid.uuid4().hex

# COMMAND ----------

# Run model training & hyper parameter tuning in parallel using hyperopt
with mlflow.start_run(run_name=uid):
  best_params = fmin(
    trials=SparkTrials(parallelism=10),
    fn = lambda params: kakapo.train_outlier_detection(params, model_space, X_train, X_test, y_test, GROUND_TRUTH_OD_EXISTS),
    space=space,
    algo=tpe.suggest,
    max_evals=50
  )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 3. Extending kakapo's default search space settings
# MAGIC 
# MAGIC Extend the default settings of kakapo and include a new model in the model space and a related selection of hyper parameter options to the search space

# COMMAND ----------

# Import a new model we want to include in our space
from pyod.models.cof import COF

# COMMAND ----------

# Enrich the default model space
model_space = kakapo.enrich_default_model_space({"cof": COF})
print("Enriched model space: {}".format(model_space))

# COMMAND ----------

additional_params = [
  {
    'type': 'cof',
    'n_neighbors': scope.int(hp.quniform('cof.n_neighbors', 5, 20, 5))
  }
]

# Enrich the default hyper param search space
search_space = kakapo.enrich_default_search_space(additional_params)
print("Enriched search space: {}".format(search_space))

# COMMAND ----------

# Load enriched search space into hyperopt
space = hp.choice('model_type', search_space)

# COMMAND ----------

# Controls wether or not we have ground truth labels to evaluate the outlier models
GROUND_TRUTH_OD_EXISTS = False 

# Unique run ID when saving MLFlow experiment
uid = uuid.uuid4().hex

# COMMAND ----------

# Run model training & hyper parameter tuning in parallel using hyperopt
with mlflow.start_run(run_name=uid):
  best_params = fmin(
    trials=SparkTrials(parallelism=10),
    fn = lambda params: kakapo.train_outlier_detection(params, model_space, X_train, X_test, None, GROUND_TRUTH_OD_EXISTS),
    space=space,
    algo=tpe.suggest,
    max_evals=50
  )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 4. Using trained models for inference
# MAGIC 
# MAGIC Load one of the trained anomaly detection models and carry out predictions on a dataset

# COMMAND ----------

# Explore the mlflow model runs
mlflow.search_runs()

# COMMAND ----------

# Use the latest run or select a custom parent run to pick out the best performing model from and the metric by which to choose the model
metric = "loss"
parentRunId = "\"" + mlflow.search_runs().iloc[0]["tags.mlflow.parentRunId"] +"\""

# Get all child runs on current experiment
runs = mlflow.search_runs(filter_string=f"tags.mlflow.parentRunId = {parentRunId}", order_by=[f"metrics.{metric} ASC"])
runs = runs.where(runs['status'] == 'FINISHED')

# Get best run id
best_run_id = runs.loc[0,'run_id']

print(best_run_id)

# COMMAND ----------

# Create a spark dataframe from the original dataset
X_test_spark_df = spark.createDataFrame(X_test.tolist(), schema = ["col1", "col2", "col3", "col4", "col5"])
X_test_spark_df.display()

# COMMAND ----------

logged_model = f'runs:/{best_run_id}/model'

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')

# Predict on a Spark DataFrame.
X_test_spark_df = X_test_spark_df.withColumn('predictions', loaded_model(struct(*map(col, X_test_spark_df.columns))))

# COMMAND ----------

# Display dataframe with the computed prediction column
X_test_spark_df.display()

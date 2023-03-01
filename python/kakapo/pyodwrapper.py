from mlflow.models.signature import infer_signature
import mlflow.pyfunc
import mlflow
from sklearn.metrics import roc_auc_score, f1_score

from .helper_methods import *

class PyodWrapper(mlflow.pyfunc.PythonModel):
    """
    MLFlow compatibility wrapper for general pyod library models supporting
    training and model predictions
    """
 
    def __init__(self, **kwargs):
      #global model_space
      #model_space = get_default_model_space()
      for key, value in kwargs.items():
        setattr(self, key, value)
          
    def set_model_space(self, model_space_input):
      #global model_space
      self.model_space = model_space_input
    
    def get_model_space(self):
      return self.model_space
        
    def fit(self, X):
      """
      Model training given input parameters and a training dataset
      """
      model_params = {k:v for k,v in self.__dict__.items() if k not in ['type', 'model_space']}#!= 'type'}
      #global model_space
      self.model = self.model_space[self.type](**model_params)
      self.model.fit(X)
  
    def evaluate(self, y_test, y_test_pred):
      """
      Evaluate model performance using roc auc
      """
      roc_auc = roc_auc_score(y_score=y_test_pred, y_true=y_test)

      return roc_auc   
 
    def predict(self, context, X):
      """
      Predict labels on provided data
      """
      result = self.model.predict(X)
      return result
    
    def decision_function(self, X):
      """
      Predict raw anomaly score of X using the fitted detector
      """
      return self.model.decision_function(X)
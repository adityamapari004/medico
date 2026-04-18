"""  train arr/test arr -> split X and y -> train multiple models ->
    evaluate each model -> select best model ->save the model pkl -> return r2 and accuracy score 
"""


import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.Linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import( RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix)
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException
from src.utils.comman import save_object,load_object


###_____configuration___

@dataclass
class ModelTrainerConfig:
    trained_model_path:str=os.path.join(
        "artifacts","model.pkl"
)
    expected_accuracy:float =0.6  ## minimum acceptable accuracy 
### ____mainClass_____

class ModelTrainer:
    def __init__(self):
        self.config=ModelTrainerConfig()

    def evaluate_model(self,X_train,y_train,X_test,y_test,models,params):
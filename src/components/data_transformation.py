""" train.csv/test.csv data transformation module
    handle the missing values, encode the categorical features, and perform feature scaling
    save preprocessor 
    return preprocessor object
    return transformed train and test data
"""

import os
import sys 
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.utils.comman import save_object

## Data Transformation Configurations
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str=os.path.join('artifacts','preprocessor.pkl')

## Data Transformation Class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self, df):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            ## identify numerical and categorical columns
            numerical_cols=df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols=df.select_dtypes(include=['object','category']).columns.tolist()

            ## remove target column from numerical and categorical columns list
            target="medicine"
            if target in numerical_cols:
                numerical_cols.remove(target)
            if target in categorical_cols:
                categorical_cols.remove(target)

            logging.info(f'Numerical columns: {numerical_cols}')
            logging.info(f'Categorical columns: {categorical_cols}')

            ## Define the numerical pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            ## Define the categorical pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder(handle_unknown='ignore')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            ## Combine numerical and categorical pipelines into a preprocessor
            preprocessor=ColumnTransformer(
                transformers=[
                    ('num_pipeline',num_pipeline,numerical_cols),
                    ('cat_pipeline',cat_pipeline,categorical_cols)
                ]
            )

            logging.info('preprocessor pipeline completed')
            return preprocessor
        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e)

    
def initiate_data_transformation(self, train_path, test_path):
        
        """ reads train and test data from the given paths, applies the data transformation steps, and returns the transformed data along with the preprocessor object path
        Args:            train_path (str): The file path to the training data CSV file.
            test_path (str): The file path to the testing data CSV file.
            reyturn X_train (numpy array): The transformed training input features.
            y_train (numpy array): The training target feature.
            X_test (numpy array): The transformed testing input features.
            y_test (numpy array): The testing target feature.
            and preprocessor_obj_file_path (str): The file path where the preprocessor object is saved.
            """
        try:
            logging.info('Initiating Data Transformation')
            # Read train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessor object')
            preprocessor_obj=self.get_data_transformer_object(train_df)
            target_column_name='medicine'
            if target_column_name not in train_df.columns:
                raise CustomException(f'Target column {target_column_name} not found in training data')

            if target_column_name not in test_df.columns:
                raise CustomException(f'Target column {target_column_name} not found in testing data')

            X_train=train_df.drop(columns=[target_column_name],axis=1)
            y_train=train_df[target_column_name]

            X_test=test_df.drop(columns=[target_column_name],axis=1)
            y_test=test_df[target_column_name]
            logging.info('Applying preprocessor object on training and testing data')
            logging.info(f'X_train columns: {X_train.shape}')
            logging.info(f'X_test columns: {X_test.shape}')
            logging.info(f'y_train columns: {y_train.shape}')
            logging.info(f'y_test columns: {y_test.shape}')

            ## encode target lable
            logging.info('Encoding target labels')
            label_encoder=LabelEncoder()
            y_train=label_encoder.fit_transform(y_train)
            y_test=label_encoder.transform(y_test)

            logging.info(f"classes found: {list(label_encoder.classes_)}")


            ## build and fit preprocessor
            preprocessor=self.get_preprocessor_object(X_train)
            logging.info('Fitting preprocessor object on training data')
            
            x_train_transformed=preprocessor.fit_transform(X_train)
            logging.info('Transforming training data')
            X_test_transformed=preprocessor.transform(X_test)
            logging.info('Transforming testing data')

            ## combine X+y arrays 

            train_arr=np.c_[x_train_transformed,y_train]
            test_arr=np.c_[X_test_transformed,y_test]


            ## save preprocessor 

            save_object(file_path=self.config.preprocessor_path,
                        obj={

                             "preprocessor":preprocessor,
                             "lableEncoder":label_encoder
                        }
                          
                        )
            logging.info("data transformation completed ")
            
            return (train_arr, test_arr,self.config.preprocessor_path)
        
        except Exception as e :
            raise CustomException (e,sys)

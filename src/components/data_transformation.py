import sys
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.exception import CustomException
from src.logger import logging
from src.utils.comman import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # 🔥 TF-IDF Pipeline
    def get_data_transformer_object(self):
        try:
            logging.info('Data transformation initiated for TEXT data')

            text_pipeline = Pipeline(steps=[
                ('tfidf', TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words='english'
                ))
            ])

            logging.info('TF-IDF pipeline created successfully')

            return text_pipeline

        except Exception as e:
            raise CustomException(e, sys)

    # 🚀 Main Transformation Function
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Reading train and test data')

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'Medicine'

            # 🔥 Convert ALL columns into single text column
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_feature_train_df = input_feature_train_df.astype(str).agg(' '.join, axis=1)

            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_df = input_feature_test_df.astype(str).agg(' '.join, axis=1)

            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying TF-IDF transformation')

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df).toarray()

            # Combine input + target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saving preprocessing object')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
import sys
import os 
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_col = ['reading_score', 'writing_score']
            categorical_col = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ])
            
            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("One_Hot_Encoder", OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ])
            logging.info("Categorical columns encoding completed")
            logging.info("Numerical columns standard scaling completed")

            preprocessor = ColumnTransformer([
                    ("num_pipeline", num_pipeline, numerical_col),
                    ("cat_pipeline", cat_pipeline, categorical_col)
                ])
            return preprocessor

        except Exception as e:
            logging.info("Error occured during Data Transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test dataset completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_obj()
            
            target_col = "math_score"
            numerical_col = ['reading_score', 'writing_score']
            
            # train df
            input_feature_train_df = train_df.drop(columns=[target_col], axis = 1)
            target_feature_train_df = train_df[target_col]

            # test df
            input_feature_test_df = test_df.drop(columns=[target_col], axis = 1)
            target_feature_test_df = test_df[target_col]

            logging.info("Start of Appliying preprocessing object on training and testing DataFrame")
            # logging.info(f"{input_feature_train_df}")

            # train df
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            # concatenate preprocessed train and test dataset with target variables
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Saved Preprocessing object")

            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path, 
                        obj = preprocessing_obj)

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)
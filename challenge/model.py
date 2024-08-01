import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from typing import Tuple, Union, List
from datetime import datetime
import numpy as np
import pickle
from pathlib import Path
import os

class DelayModel:

    FEATURES_COLS = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    def __init__(
        self
    ):
        self.path = Path(__file__).parent / "model_xgboost.pkl"
        self._model = self._load_model() # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data_process = self.process_feature(data)
        if target_column:
            features = data_process.drop(columns=[target_column])
            target = data_process[[target_column]]
            return features[self.FEATURES_COLS], target
        else:
            features = data_process
            return features[self.FEATURES_COLS]
        

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
        scale = y_train.value_counts()[0] / y_train.value_counts()[1]  # Calculate the scale_pos_weight
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        self._model.fit(x_train, y_train)
        model = self._model
        self.save_model( model)
     

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predicts delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if not self._model:
            raise ValueError("Model is not trained yet.")
        predictions = self._model.predict(features)
        return predictions.tolist()
        #return
    
    def get_min_diff(self, data):
        """
        calculates the difference of time between two var

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            The difference calculated to be asigned a column.
        """
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def process_feature(self, data):
        """
        Leaves ready the data to be processed by the model.

        Args:
            features (pd.DataFrame): preprocessed data.
        Returns:
            Data encoded.
        """
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )

        if all(col in data.columns for col in ['Fecha-O', 'Fecha-I']):
            data['min_diff'] = data.apply(self.get_min_diff, axis=1)
            threshold_in_minutes = 15
            data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
            combined_df = pd.concat([features, data['delay']], axis=1)
        else:
            combined_df = features

        
        
        for col in self.FEATURES_COLS:
            if col not in combined_df.columns:
                combined_df[col] = 0

        return combined_df



    def _load_model(self):
        """Load the model in the class var

        Returns:
            The model loaded
        """
        loaded_model = None
        if self.path.is_file():
            with open(self.path, "rb") as model_file:
                loaded_model = pickle.load(model_file)
        return loaded_model


    def save_model(self, model):
            """Save a model as pkl file.
            Args:
                model (XGBClassifier): Model to be pickled.
            """
            with open(self.path, "wb") as model_file:
                pickle.dump(model, model_file)

    
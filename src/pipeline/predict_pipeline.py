import os
import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("models", "model.pkl")
        self.preprocessor_path = os.path.join("models", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()

            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Wraps a single used-car listing's raw fields into the dataframe shape
    the preprocessor/model expect (i.e. matching the cleaned columns produced
    by DataIngestion.clean_data)."""

    def __init__(
        self,
        make: str,
        year: int,
        kilometer: int,
        fuel_type: str,
        transmission: str,
        location: str,
        color: str,
        owner: str,
        seller_type: str,
        drivetrain: str,
        seating_capacity: float,
        fuel_tank_capacity: float,
        volume: float,
        power_bhp: float,
        power_rpm: float,
        torque_nm: float,
        torque_rpm: float,
    ):
        self.make = make
        self.year = year
        self.kilometer = kilometer
        self.fuel_type = fuel_type
        self.transmission = transmission
        self.location = location
        self.color = color
        self.owner = owner
        self.seller_type = seller_type
        self.drivetrain = drivetrain
        self.seating_capacity = seating_capacity
        self.fuel_tank_capacity = fuel_tank_capacity
        self.volume = volume
        self.power_bhp = power_bhp
        self.power_rpm = power_rpm
        self.torque_nm = torque_nm
        self.torque_rpm = torque_rpm

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "Make": [self.make],
                "Year": [self.year],
                "Kilometer": [self.kilometer],
                "Fuel Type": [self.fuel_type],
                "Transmission": [self.transmission],
                "Location": [self.location],
                "Color": [self.color],
                "Owner": [self.owner],
                "Seller Type": [self.seller_type],
                "Drivetrain": [self.drivetrain],
                "Seating Capacity": [self.seating_capacity],
                "Fuel Tank Capacity": [self.fuel_tank_capacity],
                "Volume": [self.volume],
                "Power_bhp": [self.power_bhp],
                "Power_rpm": [self.power_rpm],
                "Torque_Nm": [self.torque_nm],
                "Torque_rpm": [self.torque_rpm],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

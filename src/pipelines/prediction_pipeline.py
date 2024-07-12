import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 instant:int,
                 dteday:str,
                 season:int,
                 yr:int,
                 mnth:int,
                 holiday:int,
                 weekday:int,
                 workingday:int,
                 weathersit:int,
                 temp:float,
                 atemp:float,
                 hum:float,
                 windspeed:float,
                 casual:int,
                 registered:int
                 ):
        
        self.instant = instant
        self.dteday = dteday
        self.season = season
        self.yr = yr
        self.mnth = mnth
        self.holiday = holiday
        self.weekday = weekday
        self.workingday = workingday
        self.weathersit = weathersit
        self.temp = temp
        self.atemp = atemp
        self.hum = hum
        self.windspeed = windspeed
        self.casual = casual
        self.registered = registered
        

    def get_data_as_dataframe(self):
        try:
            bike_sharing_data_dict = {
                'instant': [self.instant],
                'dteday': [self.dteday],
                'season': [self.season],
                'yr': [self.yr],
                'mnth': [self.mnth],
                'holiday': [self.holiday],
                'weekday': [self.weekday],
                'workingday': [self.workingday],
                'weathersit': [self.weathersit],
                'temp': [self.temp],
                'atemp': [self.atemp],
                'hum': [self.hum],
                'windspeed': [self.windspeed],
                'casual': [self.casual],
                'registered': [self.registered],
                
            }
            df = pd.DataFrame(bike_sharing_data_dict)
            logging.info('Dataframe gathered')
            return df
        except Exception as e:
            logging.error('Exception occurred in data gathering process')
            raise e

# Example usage:
# bike_data = BikeSharingData(1, '2011-01-01', 1, 0, 1, 0, 6, 0, 2, 0.344167, 0.363625, 0.805833, 0.160446, 331, 654, 985)
# df = bike_data.get_data_as_dataframe()

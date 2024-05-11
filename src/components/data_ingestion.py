import os 
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

## Data INgestion Config Class
@dataclass
class DataIngestionConfig():
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")
    raw_data_path = os.path.join("artifacts","raw.csv")



## Data INgestion Class

class DataIngestion():
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        
        try:
            df=pd.read_csv(os.path.join("notebooks/data","gemstone.csv"))
            logging.info("Dataset read as Pandas DataFrame")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            
            logging.info("Train Test Split")
            
            train_set,test_set=train_test_split(df,test_size=0.20,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info("Ingestion of Data is Completed")


            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
            
            
        
        except Exception as e:
            logging.info("Some Error Occured In Data Ingestion Part")
            raise CustomException(e,sys)

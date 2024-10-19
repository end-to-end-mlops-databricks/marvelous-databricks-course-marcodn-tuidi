import pandas as pd
import numpy as np
import logging

class DataProcessor:

    def __init__(self, data_path, train, config):
        self.df = self.load_data(data_path)
        self.train = train
        self.config = config
        self.X = None
        self.y = None 
        self.preprocessor = None 

    def load_data(self, data_path):
        return pd.read_csv(data_path)
    
    def create_target(self, target, raw_target):
        anlaysts = ["INTJ", "INTP", "ENTJ", "ENTP"]
        diplomats = ["INFJ", "INFP", "ENFJ", "ENFP"]
        sentinels = ["ISTJ", "ISFJ", "ESTJ", "ESFJ"]
        explorers = ["ISTP", "ISFP", "ESTP", "ESFP"]
        
        self.df[target] = (
            np.where(
                self.df[raw_target].isin(anlaysts), "Analyst",
                np.where(
                    self.df[raw_target].isin(diplomats), "Diplomat",
                    np.where(
                        self.df[raw_target].isin(sentinels), "Sentinel",
                        np.where(
                            self.df[raw_target].isin(explorers), "Explorer",
                            None
                        )
                    )
                )
            )
        )
    
    def preprocess_data(self):
        num_features = self.config['num_features']
        cat_features = self.config['cat_features']
        train_features = num_features + cat_features

        # remove columns with missing raw target
        if self.train:
            raw_target = self.config["raw_target"]
            target = self.config["target"]

            self.create_target(target, raw_target)
            self.df = self.df.dropna(subset=[target])

            self.y = self.df[target]

        self.X = self.df[train_features]

        # train features preprocessing steps


        

        


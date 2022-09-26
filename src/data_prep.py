import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

class DataPrep:
    def __init__(self, df_path):
        self.df_path = Path(df_path)
        
    def split_data(self):
        df_path = self.df_path
        df = pd.read_csv(df_path)
        y = df["name"]
        X = df.drop(columns = "name")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
        
        train_df = X_train.copy()
        train_df["name"] = y_train
        train_df["split_status"] = "train"
        
        val_df = X_val.copy()
        val_df["name"] = y_val
        val_df["split_status"] = "val"
        
        processed_df = train_df.append(val_df)
        data_directory = df_path.parent # get data directory
        processed_df.to_csv(data_directory/"images_processed.csv")
        self.processed_df = processed_df
        self.data_directory = data_directory

    def image_transfer(self):
        processed_df = self.processed_df
        data_directory = self.data_directory
        processed_data_directory = data_directory.parent
        
        # creating new file directories to hold sorted data
        if os.path.exists(processed_data_directory/"processed_data"):
            shutil.rmtree(processed_data_directory/"processed_data")
            
        for split_status in processed_df["split_status"].unique():
            os.makedirs(processed_data_directory/"processed_data"/split_status)
            for class_ in processed_df["name"].unique():
                os.makedirs(processed_data_directory/"processed_data"/split_status/class_)
                
        def image_transfer_(img_name, split_status, name):
            img_path = data_directory/"images"/(img_name + ".jpg")
            processed_img_path = processed_data_directory/"processed_data"/split_status/name/(img_name + ".jpg")
            
            shutil.copyfile(img_path, processed_img_path)
        
        processed_df.apply(lambda x: image_transfer_(img_name = x["image"], 
                                                     split_status = x["split_status"], 
                                                     name = x["name"]), 
                           axis = 1)    

def data_prep(df_path = "data/butterfly_mimics/images.csv"):
    data_prep_ = DataPrep(df_path)
    data_prep_.split_data()
    data_prep_.image_transfer()
    
if __name__ == "__main__":
    data_prep()

from concurrent.futures import process
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

class DataPrep:
    def __init__(self, df_path, test_df_path):
        self.df_path = Path(df_path)
        self.test_df_path = Path(test_df_path)
        
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
        # move images to new directory
        processed_df.apply(lambda x: self.image_transfer_(img_name = x["image"], 
                                                     split_status = x["split_status"], 
                                                     name = x["name"], 
                                                     data_directory=data_directory,
                                                     processed_data_directory=processed_data_directory), 
                           axis = 1)
        self.processed_data_directory = processed_data_directory
            
    def test_image_transfer(self):
        test_df_path = self.test_df_path
        processed_data_directory = self.processed_data_directory
        data_directory = self.data_directory
        test_df = pd.read_csv(test_df_path)
        DEFAULT_CLASS = "tiger"
        
        # if test folder already exists, delete it 
        if os.path.exists(processed_data_directory/"processed_data"/"test"):
            shutil.rmtree(processed_data_directory/"processed_data"/"test")
        
        for class_ in self.processed_df["name"].unique():
            os.makedirs(processed_data_directory/"processed_data"/"test"/class_)
        
        # move images to new directory
        test_df.apply(lambda x: self.image_transfer_(img_name = x["image"], 
                                                split_status = "test", 
                                                name = DEFAULT_CLASS, 
                                                data_directory=data_directory,
                                                processed_data_directory=processed_data_directory), 
                      axis = 1)
            
    def image_transfer_(self, img_name, split_status, name, data_directory, processed_data_directory):
        subfolder = "image_holdouts" if split_status == "test" else "images"
        
        img_path = data_directory/subfolder/(img_name + ".jpg")
        processed_img_path = processed_data_directory/"processed_data"/split_status/name/(img_name + ".jpg")
        
        shutil.copyfile(img_path, processed_img_path)

def data_prep(df_path = "data/butterfly_mimics/images.csv", 
              test_df_path = "data/butterfly_mimics/image_holdouts.csv"):
    data_prep_ = DataPrep(df_path, test_df_path)
    data_prep_.split_data()
    data_prep_.image_transfer()
    data_prep_.test_image_transfer()
    
if __name__ == "__main__":
    data_prep()

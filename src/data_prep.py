import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

class DataPrep:
    def __init__(self, df_path, test_df_path):
        """
        Args:
            df_path (str): file path to the csv file holding annotation info of train images
            test_df_path (str): file path to the csv file holding annotation info of test (or holdout) images
        """
        self.df_path = Path(df_path)
        self.test_df_path = Path(test_df_path)
        
    def split_data(self):
        """
        Splits images into train-val-test set. the processed df with train-val-test splits are then saved 
        as an attribute
        """
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
        """
        Moves images from the default file structure to one the following structure that suits mmclassification:
        |- data
            |- processed_data
                |- train
                    |- class_1
                        |- img_1
                        |- img_2
                        ...
                |- val
                    |- class_1
                        |- img_1
                        |- img_2
                        ...
                |- test
                    |- class_1
                        |- img_1
                        |- img_2
                        ...
        
        This method populates images in the train and val sets. 
        """
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
                                                     name = x["name"]), 
                           axis = 1)
        self.processed_data_directory = processed_data_directory
            
    def test_image_transfer(self):
        """
        Similar method to image_transfer, except that this method is for the test set. Since the pathing and file
        structure of the test data is different from that of train/val data, a separate method is written for the 
        test set. 
        """
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
                                                name = DEFAULT_CLASS), 
                      axis = 1)
            
    def image_transfer_(self, img_name, split_status, name):
        """
        Helper method that moves the actual file from one folder to another. This will be called by the 
        image transfer mmethods. 

        Args:
            img_name (str): file name of image 
            split_status (str): whether the image is in the train/test/val sets
            name (str): class label of image
        """
        data_directory = self.data_directory
        processed_data_directory = self.processed_data_directory
        subfolder = "image_holdouts" if split_status == "test" else "images"
        
        img_path = data_directory/subfolder/(img_name + ".jpg")
        processed_img_path = processed_data_directory/"processed_data"/split_status/name/(img_name + ".jpg")
        
        shutil.copyfile(img_path, processed_img_path)

def data_prep(df_path = "data/butterfly_mimics/images.csv", 
              test_df_path = "data/butterfly_mimics/image_holdouts.csv"):
    """
    Convenience function that instantiates the DataPrep class, then split dataset into train-val-test sets. Finally,
    move images into a folder structure that is suitable for mmclassification library.

    Args:
        df_path (str, optional): file path to the csv file holding annotation info of train images. 
        Defaults to "data/butterfly_mimics/images.csv".
        test_df_path (str, optional): file path to the csv file holding annotation info of test (or holdout) images. 
        Defaults to "data/butterfly_mimics/image_holdouts.csv".
    """
    data_prep_ = DataPrep(df_path, test_df_path)
    data_prep_.split_data()
    data_prep_.image_transfer()
    data_prep_.test_image_transfer()
    
if __name__ == "__main__":
    data_prep()

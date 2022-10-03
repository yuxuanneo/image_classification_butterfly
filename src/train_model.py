from src.mmclassification.tools.train import train_model_mmcls

from src.data_prep import data_prep

def train_model(mmcls_config_path, 
                data_path = "data/butterfly_mimics/images.csv",
                test_df_path = "data/butterfly_mimics/image_holdouts.csv",
                torch_config = None):

    # prep data file structure before training model
    data_prep(data_path)
    train_model_mmcls(mmcls_config_path)

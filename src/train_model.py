import src.train_model_torch as train_model_torch
from src.mmclassification.tools.train import train_model_mmcls

from src.data_prep import data_prep

def train_model(type, 
                mmcls_config_path, 
                data_path = "data/butterfly_mimics/images.csv",
                test_df_path = "data/butterfly_mimics/image_holdouts.csv",
                torch_config = None):

    # prep data file structure before training model
    data_prep(data_path)
    
    type = type.lower()
    assert type in ["mmcls", "torch"], "type arg only allows for 'mmcls' or 'torch'"
    
    if type == 'mmcls':
        train_model_mmcls(mmcls_config_path)

        
        
    else:
        valid_keys = ["data_dir", "batch_size", "num_epochs"]
        for key in torch_config.keys():
            assert key in valid_keys, \
                f"invalid key; torch_config only allows for the following keys: {valid_keys}" 
        train_model_torch.train_model(torch_config)
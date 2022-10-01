# from src.mmclassification.mmcls.apis.inference import init_model, inference_model
import pandas as pd
from pathlib import Path
import torch

def inference(test_df_path, config_file, checkpoint_file, 
              data_path = "data/butterfly_mimics/image_holdouts"):
    # Specify the path to model config and checkpoint file
    data_path = Path(data_path)
    config_file = config_file # 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = checkpoint_file # 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    # build the model from a config file and a checkpoint file
    model = init_model(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    test_df = pd.read_csv(test_df_path)
    images = test_df["image"].to_list()
    preds = []
    for img_name in images:
        full_img_path = data_path/(img_name +  ".jpg")
        pred = inference_model(model, str(full_img_path))
        preds.append(pred)

    preds_class = [pred["pred_class"] for pred in preds]
    test_df["name"] = preds_class
    
    return test_df

def inference_torch(model, dataloaders):
    model.eval() # set to eval mode
    all_preds = []
    images = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for (inputs, folder_idx), (file_path, _) in zip(dataloaders["test"], dataloaders["test"].dataset.imgs):
        folder_idx = folder_idx.item()
        print(input)
        print(folder_idx)
        print(file_path)
        
        class_to_idx = dataloaders["test"].dataset.class_to_idx 
        idx_to_class = {k:v for v,k in class_to_idx.items()}
        folder_name = idx_to_class[folder_idx]
        
        if folder_name == "tiger":
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds)
            
            img_name = file_path.split("/")[-1].split(".")[0]
            images.append(img_name)
    return all_preds, images
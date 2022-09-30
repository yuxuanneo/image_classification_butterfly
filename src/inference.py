from src.mmclassification.mmcls.apis.inference import init_model, inference_model
# import mmcv
import pandas as pd
from pathlib import Path

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

    return preds

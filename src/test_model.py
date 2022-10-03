from src.mmclassification.tools.test import test_model_mmcls

def test_model(mmcls_config_path, checkpoint):
    test_model_mmcls(mmcls_config_path, checkpoint)
    
    

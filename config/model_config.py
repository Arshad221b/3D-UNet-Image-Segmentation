import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Device_config = {
    'device' : device
}


Image_config = {
    'IMAGE_SIZE': 64,
    'PATCH_BATCH_SIZE': 8 
}

Model_config = {
    'BATCH_SIZE' : 1,
    'NUM_CLASS' : 15,
    'INPUT_DIM' : 1,
    'OUTPUT_CHANNEL': 15,
    'EPOCHS' : 100
}

PATHS = {
    'dataset' : '/Users/arshad_221b/Downloads/Projects/create_shorts-main/MedSeg/AMOS/amos22/',
    'model_save' : '/',
    'model_load' : '/'
    
}

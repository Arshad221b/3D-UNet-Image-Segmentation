import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Device_config = {
    'device' : device
}


Image_config = {
    'IMAGE_SIZE': 32,
}

Model_config = {
    'BATCH_SIZE' : 2,
    'NUM_CLASS' : 15,
    'INPUT_DIM' : 3,
    'OUTPUT_CHANNEL': 82,
    'EPOCHS' : 2
}

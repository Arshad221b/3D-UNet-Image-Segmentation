# 3D Medical Image Segmentation 
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project focuses on segmenting 3D MRI/CT scan images into various 15 organs. The primary architecture utilized is a custom 3D UNet, which is built upon the 2D UNet framework from 2015. Reference papers can be found in the "Papers" folder.

## Training and Validation
To run training/validation, execute `main.py`. Adjust the patch size and batch size based on the available GPU size in your system. The model was trained on an NVIDIA TITAN RTX with 24GB of RAM.

### Hyperparameters
Inside the `config` file, the following hyperparameters are defined:
* `IMAGE_SIZE`: Size of an image patch.
* `PATCH_BATCH_SIZE`: Number of patches provided to the GPU during training.
* `INPUT_DIM`: Number of channels in the input image (1 for MRI/CT, 3 for others).
* `OUTPUT_CHANNEL`: Number of output channels, i.e., the number of segments.

## Loss Function
```python
def dice_loss(input_im, target):
    smooth = 1.0
    iflat = input_im.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
```

## Other Approaches
While the main model is trained using image patches, other approaches can be explored using interpolation techniques (not recommended). Additional approaches can be found in the notebooks.


# 3D Medical Image Segmentation 

This is a project that segments 3D MRI/CT scan images into various 15 organs. The main architecture behind this is custom 3D UNet which is based on 2D UNet(2015). Reference papers are included in the Papers folder.

## Training and Validation
For training/Validation run main.py. Based on the GPU size available in the system tune the patch size and batch size for the images. The Model is trained on NVIDIA TITAN RTX with 24GB of RAM. 

### Hyperparameters 
Inside cofig following are the hyperparameters, 
* IMAGE_SIZE : Size of an image patch 
* PATCH_BATCH_SIZE : No of patches provided to the GPU at the time of training 
* INPUT_DIM : no of channels in the input image (for MRI/CT 1, for others 3)
* OUTPUT_CHANNEL : no of output channels ie no of segments

## Loss function 
```python
        def dice_loss(input_im, target):
                smooth          = 1.0
                iflat           = input_im.flatten()
                tflat           = target.flatten()
                intersection    = (iflat * tflat).sum()

                return 1 - ((2. * intersection + smooth) /
                        (iflat.sum() + tflat.sum() + smooth))
```

## Other approaches 
The main model is trained using the image patches, however other approaches can be build using interpolation techniques (not recommended). Other approaches can be found in the notebooks. 


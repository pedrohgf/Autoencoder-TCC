import pandas as pd
import numpy as np
from skimage.draw import random_shapes

def generate_mask(image):
    shaped_mask = get_shaped_mask(image)
    salted_mask = get_salt_mask(image)
    mask = shaped_mask | salted_mask
    MaskedImage = image
    MaskedImage[:,:,0] = image[:,:,0] * mask
    MaskedImage[:,:,1] = image[:,:,1] * mask
    MaskedImage[:,:,2] = image[:,:,2] * mask
    return MaskedImage

def generate_shaped_mask(image):
    shaped_mask = get_shaped_mask(image)
    MaskedImage = image
    MaskedImage[:,:,0] = image[:,:,0] * shaped_mask
    MaskedImage[:,:,1] = image[:,:,1] * shaped_mask
    MaskedImage[:,:,2] = image[:,:,2] * shaped_mask
    return shaped_mask, MaskedImage
    
def get_shaped_mask(image):
    img_size = image.shape
    Mask = random_shapes((img_size[0],img_size[1]), min_shapes=5, max_shapes=10, min_size=20, multichannel=False, allow_overlap=True)[0]
    Mask[Mask==255] = 0
    Mask[Mask>1] = 1
    return Mask

def get_salt_mask(image):
    Mask = np.zeros_like(image[:,:,0])
    for i in range(0,Mask.shape[0]):
        for j in range(0,Mask.shape[0]):
            Neighborhood = Mask[i-1:i+1,j-1]
            NumberOfFilledNeighbors = sum(Neighborhood) + Mask[i-1,j]
            Threshold = 0.95 - 0.10*NumberOfFilledNeighbors
            RandomValue = np.random.rand()
            Mask[i,j] = RandomValue > Threshold
    return Mask
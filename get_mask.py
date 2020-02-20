import pandas as pd
import numpy as np
import glob
import cv2 
from skimage.draw import random_shapes
from generate_mask import generate_shaped_mask


image_pathes = [image_path for image_path in glob.glob("train/*.png")]
#image_path_patches = image_pathes[:3]
#images = []
for image_path in image_pathes:
    image = cv2.imread(image_path)
    masked_path = image_path.replace('\\','\\img_masked_',1)
    mask_path = image_path.replace('\\','\\masked_',1)
    Mask, MaskedImage = generate_shaped_mask(image)
    cv2.imwrite(masked_path, MaskedImage)
    cv2.imwrite(mask_path, Mask)
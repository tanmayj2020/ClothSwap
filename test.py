from os import path 
import os
import numpy as np
from PIL import Image , ImageDraw
import json
import torchvision.transforms.functional as TF
from PIL import ImageDraw
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import sys

np.set_printoptions(threshold=sys.maxsize)


dataset_path_parse = path.join("./datasets" , "train" , "image-parse" , "16.png")
dataset_path_json = path.join("./datasets" , "train" , "openpose-json" , "16_keypoints.json")
dataset_path_img = path.join("./datasets" , "train" , "image" , "16.jpeg")

cloth_mask = Image.open(path.join("./datasets", "train" , "cloth-mask" ,"0.png")).convert("L")
cloth_mask = TF.resize(cloth_mask , (1024 , 768) , interpolation=InterpolationMode.NEAREST)
cloth_mask.show()
cloth_mask = np.array(cloth_mask)
cloth_mask = (cloth_mask >= 128).astype(np.uint32)
Image.fromarray(cloth_mask , "L").show()



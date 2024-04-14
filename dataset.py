import torch

import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True #fixes corrupted images

class ClassificationDataset:
    def __init__(self, image_paths, targets, resize=None, augmentations=None):
        self.image_paths=image_paths
        self.targets=targets
        self.resize=resize
        self.augmentations=augmentations
    
    def __len__(self):
        return(len(self.image_paths))
    
    def __getitem__(self, item):
        
        image=Image.open(self.image_paths[item])
        image=image.convert("RGB") #read image in proper format

        targets=self.targets[item] #extract the proper target

        # print(f"targets from dataset.py= {targets}")
        
        #resizing if need
        if self.resize is not None:
            image = image.resize((self.resize[1],self.resize[0]),  resample=Image.BILINEAR) #since it takes W,H
        
        #convert to numpy array
        image=np.array(image)

        if self.augmentations is not None:
            augmented=self.augmentations(image=image)
            image=augmented["image"]

        #since PyTorch needs CHW and we have HWC

        image=np.transpose(image, (2,0,1)).astype(np.float32)
        
        # print("targets from dataset.py=", torch.tensor(targets, dtype=torch.float))
        
        return{
            "image":torch.tensor(image, dtype=torch.float),
            "targets":torch.tensor(targets, dtype=torch.float)
        }

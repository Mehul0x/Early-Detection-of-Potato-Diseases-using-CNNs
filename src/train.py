import os
import datetime

import pandas as pd
import numpy as np

import albumentations
import torch

from sklearn import metrics
from sklearn.model_selection import train_test_split

import dataset
import engine
from model import get_model

if __name__ == "__main__":
    
    data_path=r"D:\aiml projects\Intra-Bhawan\input\ALLDATA.csv"

    device="cuda" 

    epochs=20

    df=pd.read_csv(data_path)

    images=df.img.values.tolist()
    
    #one-hot encoding the targets
    targets=pd.get_dummies(df['label'])
    targets=targets.to_numpy()
    targets=targets.astype(float)
    # targets=torch.tensor(targets)
    # targets=targets.type(torch.float)

    model= get_model(pretrained=True) #trying pre-trained

    model.to(device)#move model to device

    #since we are using a pretrained model, we need to use mean and std deviation from the imagenet dataset
    #if we do not use imagenet dataset we use mean and standard deviation of original dataset
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    #using albumentations to apply augmentations
    aug=albumentations.Compose(
        [
            albumentations.Normalize( mean, std, max_pixel_value=255.0, always_apply=True),
            # Geometric Transformations
            albumentations.HorizontalFlip(p=0.5),  # Randomly flip images horizontally
            albumentations.VerticalFlip(p=0.1),    # Randomly flip images vertically with lower probability
            albumentations.RandomRotate90(p=0.2),   # Randomly rotate images by 0, 90, 180, or 270 degrees
            albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.7),  # Random shift, scale, and rotate

            # Blur Augmentations
            albumentations.GaussianBlur(blur_limit=7, p=0.1),                                       # Apply Gaussian blur
            albumentations.MotionBlur(blur_limit=7, p=0.1)    
        ]
    )

    #using test-train-split

    train_images, valid_images, train_targets, valid_targets = train_test_split(images, targets, stratify=targets, random_state=42)

    # print(train_targets.shape)

    train_dataset=dataset.ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=(227,227),
        augmentations=aug
    )

    train_loader=torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle= False, num_workers=4
    )

    valid_dataset=dataset.ClassificationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=(227,227),
        augmentations=aug
    )

    valid_loader=torch.utils.data.DataLoader(
        valid_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    #using adam optimizer
    optimizer=torch.optim.Adam(model.parameters(), lr=5e-4)

    #training and auc score of each epoch
    for epoch in range(epochs):
        engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_targets=engine.evaluate(
            valid_loader, model, device=device
        )
        roc_auc= metrics.roc_auc_score(valid_targets, predictions)
        # blnc_acc=metrics.balanced_accuracy_score(valid_targets,predictions)
        # recall=metrics.recall_score(valid_targets, predictions)
        predictions = np.argmax(predictions, axis=1)

        # F1=metrics.f1_score(valid_targets,predictions, average="micro")
        print(
            f"Epoch={epoch}, Valid ROC AUC = {roc_auc}"
        )
    
    # Save the trained model with current date and time in the file name
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = f"D:/aiml projects/Intra-Bhawan/models/alexnet_{current_datetime}.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_save_path)
    print(f"Model saved to {model_save_path}")
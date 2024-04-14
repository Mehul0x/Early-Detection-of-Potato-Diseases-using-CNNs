# Early Detection of Potato Diseases using CNNs
This is a structured deep-learning project on CNN. By utilising transfer learning on famous existing CNN models, available on "pretrainedmodels", it is possible to create a state-of-the-art classification model for your custom dataset quickly. It uses different scripts to load the dataset, train, define the model, its loss function & backpropagation, making it look clean and easier to work with. No need to use .ipynb files to train your models, it also saves the model every time you train.

## ****Using your own dataset/making changes****



To use your dataset, you must create an "ALLDATA.csv" and provide its path in "**train.py**" . There must be two columns, 'img' containing the path to the images and another containing the labels named 'label'. The train.py script will automatically one-hot encode the labels for you. You also need to change the last output feature to the number of classes you have in model.py .

Currently, the scripts are set for Alexnet. If you want to use a different one, you have to change the **resize** parameter in the dataset creator in train.py for the specific model. If the model you are using was not trained on the ImageNet dataset you have to change the "mean" and "std", fitting for that dataset in the train.py .In the model.py you will have to change the name of the 

If you have already applied augmentations to your data, you should comment out the augmentation part of train.py

The loss function can be changed in engine.py, by default it uses BCEWithLogitsLoss

## **About the Dataset used in this**

The dataset used in this project had 15 classes with high imbalance.![image](https://github.com/Mehul0x/Early-Detection-of-Potato-Diseases-using-CNNs/assets/146676085/ee94ff09-75eb-4d13-8254-37c053650c3a)

Because of the vast imbalance, the correct way to approach this problem would be to undersample the majority classes and oversample the minority classes while performing augmentations. Due to the imbalance, F1-micro and AUC_ROC are the best scorers for this task.
The current setup gave an AUC_ROC score of 0.999 after 20 epochs
![image](https://github.com/Mehul0x/Early-Detection-of-Potato-Diseases-using-CNNs/assets/146676085/e08d3d9d-718a-433c-8593-b0327ee1219a)

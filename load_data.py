import os
import torch
import glob
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class OralCancerDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, path_to_images, path_to_csv = None, transform=None):

        # Passing the path to the train csv file reads the data from the csv with the labels
        # If None is passes insted only the images in the image folder is loaded (wich is useful for the test set)

        self.path_to_images = path_to_images
        self.path_to_csv = path_to_csv
        self.transform = transform

        if self.path_to_csv is not None:
            self.df = pd.read_csv(self.path_to_csv)

    def __len__(self):
        if self.path_to_csv:
            return len(self.df)
        else:
            return len(glob.glob(self.path_to_images + '/*.jpg'))

    def __getitem__(self, idx):
        if self.path_to_csv:
            data = self.df.iloc[idx]
            img_path = os.path.join(self.path_to_images, data['Name'])
            #image = torchvision.io.read_image(img_path)
            image = Image.open(img_path)
            label = data['Diagnosis']

            # You can input torchvision (or other) transforms and directly augment the data
            if self.transform:
                image = self.transform(image)
            # ..

            return image, label

        else:
            name = 'image_' + str(idx) + '.jpg'
            #image = torchvision.io.read_image(os.path.join(self.path_to_images, name), -1)
            image = Image.open(os.path.join(self.path_to_images, name))

            if self.transform:
                image = self.transform(image)

            return image, name
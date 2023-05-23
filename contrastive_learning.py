import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import torch.optim as optim
from pytorch_metric_learning.losses import NTXentLoss
import load_data as ld
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import glob
import pandas as pd
import os
import torchvision.models as models
from pytorch_metric_learning import losses


class CLOralCancerDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, path_to_images, path_to_csv = None, transform1=None, transform2=None):

        # Passing the path to the train csv file reads the data from the csv with the labels
        # If None is passes insted only the images in the image folder is loaded (wich is useful for the test set)

        self.path_to_images = path_to_images
        self.path_to_csv = path_to_csv
        self.transform1 = transform1
        self.transform2 = transform2

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
            if self.transform1:
                image1 = self.transform2(image)
                image2 = self.transform2(image)
            # ..

            return image1, image2

        else:
            name = 'image_' + str(idx) + '.jpg'
            #image = torchvision.io.read_image(os.path.join(self.path_to_images, name), -1)
            image = Image.open(os.path.join(self.path_to_images, name))

            if self.transform:
                image = self.transform(image)

            return image
        
#Projection head for contrastive learning
class ProjectionHead(nn.Module):
    def __init__(self, in_features=1280, hidden_features=1280, out_features=128): #size from SimCLR article
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return nn.functional.normalize(x, dim=-1)


#Classification head for classification training
class classificationHead(nn.Module):
    def __init__(self, in_features=1280, hidden_features=1280, out_features=1): #size from SimCLR article
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# Put togheter base model (efficsientnet) and projection head
class SimCLREncoder(nn.Module):
    def __init__(self, base_encoder, projection_head):
        super().__init__()
        self.encoder = nn.Sequential(
            base_encoder,
            projection_head
        )

    def forward(self, x):
        z = self.encoder(x)
        return z

# Put togheter the contrastive learned base model and the classification head
class finalModelEncoder(nn.Module):
    def __init__(self, base_encoder, class_head):
        super().__init__()
        self.encoder = nn.Sequential(
            base_encoder,
            class_head
        )

    def forward(self, x):
        z = self.encoder(x)
        return z

    def predict(self, x):
        z = self.encoder(x)
        return torch.sigmoid(z)

def augmentImage(images):
    #TODO
    transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.5, 3)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()
    ])
    augment_image = transform(images)
    return augment_image

def plotImages(dataset, num_images):
    '''Takes a tensor of images and plots the 8 first'''
    #TODO
    batch = next(iter(dataset))
    #print(batch.type)
    batch_np = batch[0].numpy()
    #indices = np.random.choice(batch_np.shape[0], size=num_images, replace=False)
    images = batch_np[0:8]

    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    # Iterate over the images and display them
    for i, ax in enumerate(axs.flat):
        image = np.transpose(images[i], (1, 2, 0))
        ax.imshow(image)
        ax.axis('off')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

#Data path
path_to_csv = '/Users/falk/Documents/DL4IA/DL4IA_A5/cancer-classification-challenge-2023/train.csv'
path_to_train_images = '/Users/falk/Documents/DL4IA/DL4IA_A5/cancer-classification-challenge-2023/train'
path_to_test_images = '/Users/falk/Documents/DL4IA/DL4IA_A5/cancer-classification-challenge-2023/test'

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

#TODO check if cancer images have similar mean and std
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

transform_aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

train = CLOralCancerDataset(path_to_train_images, path_to_csv, transform1=transform, transform2=transform_aug )


train_dataloader = DataLoader(train,
    batch_size=100,
    shuffle=False)

# Load the pre-trained EfficientNet model
resnet50 = models.resnet50(pretrained=True)
resnet50 = nn.Sequential(*list(resnet50.children())[:-2]) 


model_name = 'efficientnet-b0'
base_model = EfficientNet.from_pretrained(model_name)

#TODO understand the input needed for the loss function, match with the forward pass in the CL training loop
loss_func = NTXentLoss()

# Replace the classifier with a identity operation, replace with projection head.
base_model._fc = nn.Identity()

projection_head = ProjectionHead()
class_head = classificationHead()

# Build the SimCLR model with the EfficientNet base and projection head
simclr_model = SimCLREncoder(base_model, projection_head)
final_model = finalModelEncoder(base_model, class_head)

optimizerCL = optim.Adam(simclr_model.parameters(), lr=0.001)



# Train the model
num_epochs = 15
for epoch in range(num_epochs):
    print("start training epoch:", epoch+1)
    simclr_model.train()
    running_loss = 0.0
    for images1, images2 in train_dataloader:
        print(type(images1))
        result = torch.stack((images1, images2), dim=1).view(-1, 3, 128, 128)
        result = result.to(device)
        # Zero the gradients
        optimizerCL.zero_grad()

        # Forward pass
        batch_size = result.size(0)
        embeddings = simclr_model(result)
        # The assumption here is that data[0] and data[1] are a positive pair
        # data[2] and data[3] are the next positive pair, and so on
        #labels = torch.arange(batch_size)
        #labels[1::2] = labels[0::2]
        N = batch_size/2
        labels = 2 * (torch.arange(2*N) % N)
        labels = labels.to(device)
        loss = loss_func(embeddings, labels)
        loss.backward()
        print(loss)




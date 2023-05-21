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
path_to_csv = '/Users/youdengsong/Desktop/DL_project/cancer-classification-challenge-2023/train.csv'
path_to_train_images = '/Users/youdengsong/Desktop/DL_project/cancer-classification-challenge-2023/train'
path_to_test_images = '/Users/youdengsong/Desktop/DL_project/cancer-classification-challenge-2023/test'

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = ld.OralCancerDataset(path_to_train_images, path_to_csv, transform=transform)
test_dataset = ld.OralCancerDataset(path_to_test_images, None, transform=transform)
aug_dataset = augmentImage(train_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
aug_dataloader = DataLoader(aug_dataset, batch_size=32, shuffle=False)

plotImages(aug_dataloader, 8)


# # Load the pre-trained EfficientNet model
# model_name = 'efficientnet-b0'
# base_model = EfficientNet.from_pretrained(model_name)

# #TODO understand the input needed for the loss function, match with the forward pass in the CL training loop
# loss_func = NTXentLoss()

# # Replace the classifier with a identity operation, replace with projection head.
# base_model._fc = nn.Identity()

# projection_head = ProjectionHead()
# class_head = classificationHead()

# # Build the SimCLR model with the EfficientNet base and projection head
# simclr_model = SimCLREncoder(base_model, projection_head)
# final_model = finalModelEncoder(base_model, class_head)

# # Create normalizer for input 
# #TODO check if cancer images have similar mean and std
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# optimizerCL = optim.Adam(simclr_model.parameters(), lr=0.001)


# x1 = torch.randn(100, 3, 128, 128)  # a dummy input of the correct size


# normalized_x1 = normalize(x1)

# #TODO training CL model loop
# for i in range(1):
#     z1 = simclr_model(normalized_x1)
#     #loss = loss_func(temperature=0.07, **kwargs)
#     #optimizerCL.zero_grad()
#     #loss.backward()
#     #optimizerCL.step()
#     print(z1.shape)
# z3 = final_model.predict(normalized_x1)


# #TODO training classification model loop
# for j in range(1):
#     pass





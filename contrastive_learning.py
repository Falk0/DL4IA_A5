import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import torch.optim as optim
from pytorch_metric_learning.losses import NTXentLoss


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
    pass
    return 

def plotImages(images):
    '''Takes a tensor of images and plots the 8 first'''
    #TODO
    pass
    return 

# Load the pre-trained EfficientNet model
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

# Create normalizer for input 
#TODO check if cancer images have similar mean and std
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

optimizerCL = optim.Adam(simclr_model.parameters(), lr=0.001)


x1 = torch.randn(100, 3, 128, 128)  # a dummy input of the correct size


normalized_x1 = normalize(x1)

#TODO training CL model loop
for i in range(1):
    z1 = simclr_model(normalized_x1)
    #loss = loss_func(temperature=0.07, **kwargs)
    #optimizerCL.zero_grad()
    #loss.backward()
    #optimizerCL.step()
    print(z1.shape)
z3 = final_model.predict(normalized_x1)


#TODO training classification model loop
for j in range(1):
    pass





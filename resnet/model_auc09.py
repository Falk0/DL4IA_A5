import os
import torch
import glob
import torchvision
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())

angle_range = (-45, 45)

transform = transforms.Compose([
    #transforms.ColorJitter(brightness=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(angle_range),
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.5, 3)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[100.6899, 125.9274, 141.6127], std=[39.1786, 35.4815, 29.5352])  # ImageNet normalization
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[100.6899, 125.9274, 141.6127], std=[39.1786, 35.4815, 29.5352])  # ImageNet normalization
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[102.3741, 129.7284, 145.4113], std=[40.5789, 37.1226, 30.4842])  # ImageNet normalization
])

class OralCancerDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, path_to_images, path_to_csv = None, transform=None, train=True):

        # Passing the path to the train csv file reads the data from the csv with the labels
        # If None is passes insted only the images in the image folder is loaded (wich is useful for the test set)

        self.path_to_images = path_to_images
        self.path_to_csv = path_to_csv
        self.transform = transform
        self.train = train
        self.length = 0

        if self.path_to_csv is not None:
            self.df = pd.read_csv(self.path_to_csv)
            if self.train:
                #self.df = pd.read_csv(self.path_to_csv)
                data = pd.read_csv(self.path_to_csv)
                data.set_index("Name", inplace = True)
                train_patient = ("pat_009", "pat_096", "pat_086", "pat_025", "pat_067", "pat_077", "pat_071", "pat_063")
                self.df = data[data.index.str.startswith(train_patient)]
                self.df.reset_index(inplace=True)
                self.length = len(self.df)
            else:
                data = pd.read_csv(self.path_to_csv)
                data.set_index("Name", inplace = True)
                val_patient = ("pat_053", "pat_081")
                self.df = data[data.index.str.startswith(val_patient)]
                self.df.reset_index(inplace=True)
                self.length = len(self.df)

    def __len__(self):
        if self.path_to_csv:
            return self.length
        else:
            return len(glob.glob(self.path_to_images + '/*.jpg'))

    def __getitem__(self, idx):
        if self.path_to_csv:
            data = self.df.iloc[idx]
            img_path = os.path.join(self.path_to_images, data['Name'])
            image = Image.open(img_path)
            label = data['Diagnosis']
            # You can input torchvision (or other) transforms and directly augment the data
            if self.transform:
                image = self.transform(image)
            
            return image, label
            # 

        else:
            name = 'image_' + str(idx) + '.jpg'
            #image = torchvision.io.read_image(os.path.join(self.path_to_images, name), -1)
            image = Image.open(os.path.join(self.path_to_images, name))

            if self.transform:
                image = self.transform(image)

            return image, name

path_to_csv = '/kaggle/input/cancer-classification-challenge-2023/train.csv'
path_to_train_images = '/kaggle/input/cancer-classification-challenge-2023/train'
path_to_test_images = '/kaggle/input/cancer-classification-challenge-2023/test'

train_dataset = OralCancerDataset(path_to_train_images, path_to_csv, transform=transform, train=True)
val_dataset = OralCancerDataset(path_to_train_images, path_to_csv, transform=transform_val, train=False)
test_dataset = OralCancerDataset(path_to_test_images, None, transform_test)


#generator1 = torch.Generator().manual_seed(42)
#train_set, val_set = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.8), int(len(train_dataset)) - int(len(train_dataset)*0.8)], generator=generator1)

#train_set.dataset.transform = transform
#val_set.dataset.transform = transform


train_dataloader = DataLoader(train_dataset,
    batch_size=32,
    shuffle=True)

val_dataloader = DataLoader(val_dataset,
    batch_size=32,
    shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the ResNet-50 model
model = torchvision.models.resnet50(weights=True)

for param in model.parameters():
    param.requires_grad = False

# Find the total number of layers in the model
total_layers = len(list(model.children()))

# Calculate the index to split the layers
split_index = total_layers // 2

# Iterate over the model's children and unfreeze the parameters after the split index
for i, child in enumerate(model.children()):
    if i >= split_index + 1:
        for param in child.parameters():
            param.requires_grad = True


for name, param in model.named_parameters():
    print(name, param.requires_grad)

# Modify the last fully connected layer
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
# Create a StepLR scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Define the regularization strength
l2_lambda = 0.01

# # Calculate the L2 regularization loss
# l2_regularization = torch.tensor(0.)
# for param in model.parameters():
#     l2_regularization += torch.norm(param, p=2)

train_loss = []
train_auc = []
val_loss = []
val_auc = []
epochs = []

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    print("start training epoch:", epoch+1)
    model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images.float())

        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.tolist())
        #y_pred.extend(predicted.tolist())
        y_pred.extend(F.softmax(outputs, dim=1)[:, 1].tolist())  # Use softmax and extract the probability of class 1
        # Calculate the L2 regularization loss
        l2_regularization = torch.tensor(0.)
        l2_regularization = l2_regularization.to(device)
        for param in model.parameters():
            l2_regularization += torch.norm(param, p=2)
        
        # Calculate the F1 score

        loss = criterion(outputs, labels) + l2_lambda * l2_regularization

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the average loss for the epoch
    scheduler.step()
    train_loss.append(running_loss/len(train_dataloader))
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader):.4f}")
#     f1 = f1_score(y_true, y_pred, average='macro')
#     print(f"Train F1 Score: {f1:.4f}")
#     F1_score_train.append(f1)
    # Calculate AUC score
    train_auc_score = roc_auc_score(y_true, y_pred)
    print(f"Train AUC Score: {train_auc_score:.4f}")
    train_auc.append(train_auc_score)

    # Validation
    model.eval()
    y_true = []
    y_pred = []
    running_loss = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images.float())
            #loss = criterion(outputs, labels)
            #running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            #y_pred.extend(predicted.tolist())
            y_pred.extend(F.softmax(outputs, dim=1)[:, 1].tolist())  # Use softmax and extract the probability of class 1
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    val_loss.append(running_loss/len(val_dataloader))
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(val_dataloader):.4f}")
    # Calculate AUC score for validation set
    val_auc_score = roc_auc_score(y_true, y_pred)
    print(f"Val AUC Score: {val_auc_score:.4f}")
    val_auc.append(val_auc_score)
    epochs.append(epoch+1)

d = {'Name':[], 'Diagnosis':[]}
for images, name in test_dataloader:
    images = images.to(device)
    outputs = model(images.float())
    #_, predicted = torch.max(outputs.data, 1)
    probabilities = F.softmax(outputs, dim=1)
    #label = predicted.tolist()
    d['Name'].extend(name)
    #d['Diagnosis'].extend(label)
    d['Diagnosis'].extend(probabilities[:, 1].tolist())

df = pd.DataFrame(d)
df.to_csv('submission.csv', index = False)


fig1, (ax1, ax2) = plt.subplots(1, 2)
fig1.set_figwidth(10)

ax1.set_title('loss')
ax1.set_xlabel('Epochs')
ax1.plot(epochs, train_loss, label='training loss')
ax1.plot(epochs, val_loss, label='val loss')
ax1.grid()
ax1.legend()

ax2.set_title('AUC score')
ax2.set_xlabel('Epochs')
ax2.plot(epochs, train_auc, label= 'Train F1')
ax2.plot(epochs, val_auc, label= 'Val F1')
ax2.grid()
ax2.legend()
plt.show()
plt.savefig('Train_graph.png', dpi = 200)
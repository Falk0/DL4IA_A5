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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

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

path_to_csv = 'train.csv'
path_to_train_images = 'train'
path_to_test_images = 'test'

train_dataset = OralCancerDataset(path_to_train_images, path_to_csv)

test_dataset = OralCancerDataset(path_to_test_images, None, transform_test)


generator1 = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.8), int(len(train_dataset)) - int(len(train_dataset)*0.8)], generator=generator1)

train_set.dataset.transform = transform
val_set.dataset.transform = transform


train_dataloader = DataLoader(train_set,
    batch_size=32,
    shuffle=True)

val_dataloader = DataLoader(val_set,
    batch_size=32,
    shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the ResNet-50 model
model = torchvision.models.resnet50()

# Modify the last fully connected layer
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss = []
F1_score_train = []
val_loss = []
F1_score_val = []
epochs = []

# Train the model
num_epochs = 15
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
        y_pred.extend(predicted.tolist())

        # Calculate the F1 score

        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the average loss for the epoch
    train_loss.append(running_loss/len(train_dataloader))
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader):.4f}")
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Train F1 Score: {f1:.4f}")
    F1_score_train.append(f1)

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
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
    
    val_loss.append(running_loss/len(val_dataloader))
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(val_dataloader):.4f}")
    # Calculate the F1 score
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Validation F1 Score: {f1:.4f}")
    F1_score_val.append(f1)
    epochs.append(epoch+1)

d = {'Name':[], 'Diagnosis':[]}
for images, name in test_dataloader:
    images = images.to(device)
    outputs = model(images.float())
    _, predicted = torch.max(outputs.data, 1)
    label = predicted.tolist()
    d['Name'].extend(name)
    d['Diagnosis'].extend(label)

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

ax2.set_title('F1 score')
ax2.set_xlabel('Epochs')
ax2.plot(epochs, F1_score_train, label= 'Train F1')
ax2.plot(epochs, F1_score_val, label= 'Val F1')
ax2.grid()
ax2.legend()
plt.savefig('Train_graph.png', dpi = 200)

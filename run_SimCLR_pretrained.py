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
import os
import os.path
import random
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the model
simclr_model = torch.load('C:/Users/linus/python_projects/deep_learning_for_image_analysis/data/CL_pretrained_model_2.pt')
#simclr_model = torchvision.models.resnet50(pretrained = True) 
simclr_model.fc = nn.Linear(2048,2)

#testing with two layer, remove if not effective
#simclr_model.fc = nn.Sequential(
#    nn.Linear(512, 256),
##    nn.ReLU(inplace=True),
#    nn.Linear(256, 2)
#)

print(simclr_model)


for name, param in simclr_model.named_parameters():
    param.requires_grad = False
    

blocks = [
    ['layer4', 'fc'],
    ['layer3', 'layer4', 'fc'],
    ['layer2', 'layer3', 'layer4', 'fc'],
]


simclr_model = simclr_model.to(device)


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

class CustomRotation:
    def __init__(self, degrees, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            degree = random.choice(self.degrees)
            return transforms.functional.rotate(img, degree)
        return img

path_to_csv_train = 'C:/Users/linus/python_projects/deep_learning_for_image_analysis/data/class_train/train.csv'
path_to_csv_val = 'C:/Users/linus/python_projects/deep_learning_for_image_analysis/data/class_val/val.csv'
path_to_train_images = 'C:/Users/linus/python_projects/deep_learning_for_image_analysis/data/class_train'
path_to_val_images = 'C:/Users/linus/python_projects/deep_learning_for_image_analysis/data/class_val'
path_to_test_images = 'C:/Users/linus/python_projects/deep_learning_for_image_analysis/data/test'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([ transforms.GaussianBlur(kernel_size=9)],p=0.5),
    CustomRotation([90, 180, 270], p=0.5),
    transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.1, 
                                                                     contrast=0.1, 
                                                                     saturation=0.1, 
                                                                     hue=0.1)
                                          ], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
])


train_set = OralCancerDataset(path_to_train_images, path_to_csv_train, transform_test)
val_set = OralCancerDataset(path_to_val_images, path_to_csv_val, transform_test)
test_dataset = OralCancerDataset(path_to_test_images, None, transform_test)


train_dataloader = DataLoader(train_set,
    batch_size=16,
    shuffle=True)

val_dataloader = DataLoader(val_set,
    batch_size=16,
    shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

####

initial_lr = 5e-4
step_size = 3  # Decrease the learning rate every 3 epochs
gamma = 0.8   # Decrease the learning rate to 10% of the previous rate

# Initialize your optimizer
optimizer = optim.Adam(simclr_model.parameters(), lr=initial_lr, weight_decay=1e-3)

# Initialize your learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

####

weights = [0.33, 0.67]
class_weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)


train_loss = []
F1_score_train = []
val_loss = []
F1_score_val = []
epochs = []

best_val_f1 = 0.0 


# Train the model
num_epochs = 15
for epoch in range(num_epochs):
    print("start training epoch:", epoch+1)
    simclr_model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []
    
    #unfreeze layers after some amount of training

    if epoch < len(blocks):
        print(blocks[epoch])
        for name, param in simclr_model.named_parameters():
            if any(block in name for block in blocks[epoch]):
                param.requires_grad = True

    #if epoch == 5:
    #    for name, param in simclr_model.named_parameters():
    #        param.requires_grad = True

    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = simclr_model(images.float())

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

    #step the learning rate
    scheduler.step()

    # Validation
    simclr_model.eval()
    y_true = []
    y_pred = []
    running_loss = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = simclr_model(images.float())
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
    if f1 > best_val_f1:
        best_val_f1 = f1
        torch.save(simclr_model.state_dict(), 'best_model_state_dict.pt')

simclr_model.load_state_dict(torch.load('best_model_state_dict.pt'))



d = {'Name':[], 'Diagnosis':[]}
for images, name in test_dataloader:
    images = images.to(device)
    outputs = simclr_model(images.float())
    
    # Apply softmax to output
    probabilities = F.softmax(outputs, dim=1).detach().cpu().numpy()
    
    # Select probability of positive class (assuming it corresponds to the second output unit)
    pos_probabilities = probabilities[:, 1]
    
    d['Name'].extend(name)
    for prob in pos_probabilities:
        d['Diagnosis'].append(float(prob))  #

df = pd.DataFrame(d)
df.to_csv('submission.csv', index = False)

# Plotting graphs
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

print(best_val_f1)

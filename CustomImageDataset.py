from torchvision.datasets import VisionDataset
from PIL import Image
import os
import os.path

class CustomImageDataset(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(CustomImageDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.images = []
        self.labels = []
        
        split_dir = os.path.join(root, split)
        
        if split != 'unlabeled':
            for label, d in enumerate(os.listdir(split_dir)):
                directory = os.path.join(split_dir, d)
                if os.path.isdir(directory):
                    for img in os.listdir(directory):
                        if img.endswith('.jpg'):
                            self.images.append(os.path.join(directory, img))
                            self.labels.append(label)
        else:  # no labels for 'unlabeled' split
            for img in os.listdir(split_dir):
                if img.endswith('.jpg'):
                    self.images.append(os.path.join(split_dir, img))
                    self.labels.append(-1)  # use -1 or any other distinctive value for unlabeled data

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
            return len(self.images)
import torch
from torchvision import transforms
from PIL import Image
import os
import glob
from torch.utils.data import Dataset

def load4CL():
    # Directory containing your images
    directory = '/Users/falk/Documents/DL4IA/DL4IA_A5/cancer-classification-challenge-2023/mini_train/'

    # Define your transformations
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Load all images from the directory
    image_paths = glob.glob(os.path.join(directory, '*.jpg'))
    # Create a list to store your images
    images = []

    # Load each image, create an augmented version and save both to the list
    for image_path in image_paths:
        # Load the image
        image = Image.open(image_path)
        images.append(transforms.ToTensor()(image))

        # Apply the transformations to create an augmented version
        augmented_image = transform(image)
        images.append(augmented_image)

    # Convert the list of images to a single tensor
    tensor = torch.stack(images)

    # Save the tensor to a file
    return tensor
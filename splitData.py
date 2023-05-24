patient = ['096', '086', '081', '077', '071', '067', '063', '053', '025', '009']

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
                train_patient = ("pat_009", "pat_096", "pat_086", "pat_025", "pat_067", "pat_077", "pat_071")
                self.df = data[data.index.str.startswith(train_patient)]
                self.df.reset_index(inplace=True)
                self.length = len(self.df)
            else:
                data = pd.read_csv(self.path_to_csv)
                data.set_index("Name", inplace = True)
                val_patient = ("pat_053", "pat_063", "pat_081")
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

path_to_csv = 'train.csv'
path_to_train_images = 'train'
path_to_test_images = 'test'

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[100.6899, 125.9274, 141.6127], std=[39.1786, 35.4815, 29.5352])
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[100.6899, 125.9274, 141.6127], std=[39.1786, 35.4815, 29.5352])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[102.3741, 129.7284, 145.4113], std=[40.5789, 37.1226, 30.4842])
])


train_dataset = OralCancerDataset(path_to_train_images, path_to_csv, transform=transform_train, train=True)
val_dataset = OralCancerDataset(path_to_train_images, path_to_csv, transform=transform, train=False)
test_dataset = OralCancerDataset(path_to_test_images, None, transform_test)

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SegmentationDataset(Dataset):
    """
    Segmentation dataset
    """
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Root directory
        :param transform: Transformations to apply
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        try:
            self.classes.remove(".DS_Store")
        except:
            pass
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self.load_images()

    def load_images(self):
        """
        Load images
        :return: List of images
        """
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                images.append((image_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        """
        :return: Length of dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        :param index: Index
        :return: Image and label
        """
        image_path, label = self.images[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label

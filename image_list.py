import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import DatasetFolder, default_loader
import numpy as np
import random
import pdb

def load_image(img_path):
    img = Image.open(img_path)
    img = img.convert("RGB")
    return img


class ImageList(Dataset):
    def __init__(
        self,
        image_root: str,
        label_file: str,
        transform=None,
        pseudo_item_list=None,
    ):
        self.image_root = image_root
        self._label_file = label_file
        self.transform = transform

        assert (
            label_file or pseudo_item_list
        ), f"Must provide either label file or pseudo labels."
        self.item_list = (
            self.build_index(label_file) if label_file else pseudo_item_list
        )

    def build_index(self, label_file):
        """Build a list of <image path, class label> items.

        Args:
            label_file: path to the domain-net label file

        Returns:
            item_list: a list of <image path, class label> items.
        """
        # read in items; each item takes one line
        with open(label_file, "r") as fd:
            lines = fd.readlines()
        lines = [line.strip() for line in lines if line]

        item_list = []
        for item in lines:
            img_file, label = item.split()
            img_path = os.path.join(self.image_root, img_file)
            label = int(label)
            item_list.append((img_path, label))

        return item_list

    def __getitem__(self, idx):
        """Retrieve data for one item.

        Args:
            idx: index of the dataset item.
        Returns:
            img: <C, H, W> tensor of an image
            label: int or <C, > tensor, the corresponding class label. when using raw label
                file return int, when using pseudo label list return <C, > tensor.
        """
        img_path, label = self.item_list[idx]
        img = load_image(img_path)
        if self.transform:
            img = self.transform(img)

        return img, label, idx

    def __len__(self):
        return len(self.item_list)

class ImageList_Percent(Dataset):
    def __init__(
        self,
        image_root: str,
        label_file: str,
        percent=1,
        random_seed=0,
        transform=None,
        pseudo_item_list=None,
    ):
        self.image_root = image_root
        self._label_file = label_file
        self.percent = percent
        self.random_seed = random_seed
        self.transform = transform

        assert (
            label_file or pseudo_item_list
        ), f"Must provide either label file or pseudo labels."
        self.item_list = (
            self.build_index(label_file) if label_file else pseudo_item_list
        )

    def build_index(self, label_file):
        """Build a list of <image path, class label> items.

        Args:
            label_file: path to the domain-net label file

        Returns:
            item_list: a list of <image path, class label> items.
        """
        # read in items; each item takes one line
        with open(label_file, "r") as fd:
            lines = fd.readlines()
        lines = [line.strip() for line in lines if line]
        
        img_list = []
        lab_list = []

        for item in lines:
            img_file, label = item.split()
            img_path = os.path.join(self.image_root, img_file)
            label = int(label)
            img_list.append(img_path)
            lab_list.append(label)
        
        num_class = len(np.unique(np.array(lab_list)))
        img_cls_list = []
        for i in range(num_class):
            img_cls_list.append([])

        for i in range(len(img_list)):
            # images of the same class are in the same sub list
            lab = lab_list[i]
            img_cls_list[lab].append(img_list[i])

        item_img_list = []
        item_lab_list = []
        for i in range(num_class):
            print(f"Class{i} has {len(img_cls_list[i])} samples.")
            num_sample = int(len(img_cls_list[i]) * self.percent)
            random.Random(self.random_seed).shuffle(img_cls_list[i])
            
            item_img_list.extend(img_cls_list[i][:num_sample])
            item_lab_list.extend([i] * num_sample)
        
        assert len(item_img_list) == len(item_lab_list)
        item_list = list(zip(item_img_list, item_lab_list))
    
        return item_list


    def __getitem__(self, idx):
        """Retrieve data for one item.

        Args:
            idx: index of the dataset item.
        Returns:
            img: <C, H, W> tensor of an image
            label: int or <C, > tensor, the corresponding class label. when using raw label
                file return int, when using pseudo label list return <C, > tensor.
        """
        img_path, label = self.item_list[idx]
        img = load_image(img_path)
        if self.transform:
            img = self.transform(img)

        return img, label, idx

    def __len__(self):
        return len(self.item_list)



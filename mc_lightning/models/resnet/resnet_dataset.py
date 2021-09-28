import torch
import torch.utils.data as data
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm 
from PIL import Image
from mc_lightning.utilities.utilities import pil_loader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
from skimage import color

# import staintools
# from p_tqdm import p_umap
import warnings

class SlideDataset(data.Dataset):
    """
    Modification of vanilla `tmb_bot.utilities.Dataset` class to facilitate having
    a label for classification as well as the slide name itself
    """
    def __init__(self, paths, slide_ids, labels, transform_compose, bw = 'None'):
        """
        Paths and labels should be array like
        """
        self.paths = paths
        self.slide_ids = slide_ids
        self.labels = labels
        self.transform = transform_compose
        self.bw = bw

    def __len__(self):
        'Denotes the total number of samples'
        return self.paths.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'

        img_path = self.paths[index]
        pil_file = pil_loader(img_path, self.bw)
        pil_file = self.transform(pil_file)
        slide_id = self.slide_ids[index]
        label = self.labels[index]

        return pil_file, label, slide_id

class ContrastiveSlideDataset(data.Dataset):
    """
    Modification of vanilla `tmb_bot.utilities.Dataset` class to facilitate having
    a label for classification as well as the slide name itself
    """
    def __init__(self, paths1, paths2, slide_ids1, slide_ids2, labels, transform_compose):
        """
        Paths and labels should be array like
        """
        self.paths1 = paths1
        self.paths2 = paths2

        self.slide_ids1 = slide_ids1
        self.slide_ids2 = slide_ids2

        self.labels = labels
        
        self.transform = transform_compose

    def __len__(self):
        'Denotes the total number of samples'
        return self.paths1.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'

        img_path1 = self.paths1[index]
        img_path2 = self.paths2[index]

        pil_file1 = pil_loader(img_path1)
        pil_file2 = pil_loader(img_path2)

        pil_file1 = self.transform(pil_file1)
        pil_file2 = self.transform(pil_file2)

        slide_id1 = self.slide_ids1[index]
        slide_id2 = self.slide_ids2[index]

        label = self.labels[index]

        return pil_file1, pil_file2, label, slide_id1, slide_id2

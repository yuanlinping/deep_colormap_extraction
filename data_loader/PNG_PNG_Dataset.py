from .BaseDataset import BaseDataset
import torch
import numpy as np
import cv2

import logging
import time

logger = logging.getLogger('PNG_PNG_Dataset')
logger.setLevel(logging.DEBUG)
logger.disabled = True

class PNG_PNG_Dataset(BaseDataset):
    def __init__(self,file_list="./dataset/train.txt",
                image_paras = {'width':256,'height':128,'channel':3},
                label_paras ={'width':256,'height':10,'channel':3},
                color_space = "Lab",
                 is_label_normalized = True):
        super(PNG_PNG_Dataset,self).__init__(file_list, image_paras, label_paras)
        self.color_space = color_space
        self.is_label_normalized = is_label_normalized

    def __getitem__(self, idx):
        image_file_path = self.image_paths[idx]
        label_file_path = self.label_paths[idx]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('{}\t{}'.format(image_file_path, label_file_path))

        image = cv2.imread(image_file_path)
        label = cv2.imread(label_file_path)

        # convert label to a specific color space
        if (self.color_space == "Lab"):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2Lab)
        elif (self.color_space == 'Rgb'):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        image = image.reshape(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNEL)
        label = label.reshape(self.LABEL_HEIGHT, self.LABEL_WIDTH, self.LABEL_CHANNEL)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        image = image.permute(2,0,1) / 255.
        if self.is_label_normalized:
            label = label.permute(2,0,1) / 255.
        else:
            label = label.permute(2,0,1)

        if (label != label).any():
            logger.debug("there are nan in label")
        if (image != image).any():
            logger.debug("there are nan in image")

        return {'image': image, 'label': label}


if __name__=='__main__':
    from torch.utils.data import DataLoader
    from boxx import loga
    from matplotlib import pyplot as plt

    dataset = PNG_PNG_Dataset(file_list="./dataset/train.txt")
    dataloader = DataLoader(dataset, batch_size=1)

    for ind, batch in enumerate(dataloader):
        if (ind == 0):
            image = batch['image']
            label = batch['label']
            label = label.numpy()
            label = np.transpose(label[0], (1,2,0)) * 255.
            label = label.astype(np.uint8)
            label = cv2.cvtColor(label, cv2.COLOR_Lab2RGB)
            plt.figure()
            plt.imshow(label)
            plt.show()
        else:
            break



from .BaseDataset import BaseDataset
import torch
import numpy as np
import cv2

import logging
import time

logger = logging.getLogger('CSV_PNG_Dataset')
logger.setLevel(logging.DEBUG)
logger.disabled = True

class CSV_PNG_Dataset(BaseDataset):
    def __init__(self,file_list="./dataset/train.txt",
                image_paras = {'width':64,'height':64,'channel':64},
                label_paras ={'width':256,'height':10,'channel':3},
                color_space = "Lab",
                is_label_normalized = True):
        super(CSV_PNG_Dataset,self).__init__(file_list, image_paras, label_paras)
        self.color_space = color_space
        self.is_label_normalized = is_label_normalized

    def __getitem__(self, idx):
        image_file_path = self.image_paths[idx]
        label_file_path = self.label_paths[idx]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('{}\t{}'.format(image_file_path, label_file_path))

        image = np.genfromtxt(image_file_path, delimiter=',')
        label = cv2.imread(label_file_path)

        # convert label to a specific color space
        if (self.color_space == "Lab"):
            label = cv2.cvtColor(label, cv2.COLOR_BGR2Lab) #after conversion, the range of values of three channels are within [0,255]

        if logger.isEnabledFor(logging.DEBUG):
            tm = time.time()
            logger.debug('\n\n')
            logger.debug("start to preprocessing {}".format(tm - tm))

        image = image.reshape(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNEL)  # 3D image histogram is [light][a][b]
        label = label.reshape(self.LABEL_HEIGHT, self.LABEL_WIDTH, self.LABEL_CHANNEL)

        if logger.isEnabledFor(logging.DEBUG):
            tm = time.time() - tm
            logger.debug("after reshape the data, time eclipsed {}".format(tm))

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()


        if logger.isEnabledFor(logging.DEBUG):
            tm = time.time() - tm
            logger.debug("\n finishing change numpy to tensor, time eclipsed:{}".format(tm))

        if self.color_space == "Lab":
            image[0][int(self.IMAGE_WIDTH / 2)][int(self.IMAGE_CHANNEL / 2)] = 0.0  # black   [lightness][a][b]
            image[int(self.IMAGE_HEIGHT - 1)][int(self.IMAGE_WIDTH / 2)][int(self.IMAGE_CHANNEL / 2)] = 0.0 # white

        maxV = torch.max(image)
        minV = torch.min(image)
        diffV = (maxV - minV) * 1.0
        image = (image - minV) / diffV

        image = image.permute(2,0,1)
        if self.is_label_normalized:
            label = label.permute(2,0,1) / 255.
        else:
            label = label.permute(2,0,1)

        if logger.isEnabledFor(logging.DEBUG):
            tm = time.time() - tm
            logger.debug("\n after transpose time eclipsed:{}".format(tm))

        if (label != label).any():
            logger.debug("there are nan in label")
        if (image != image).any():
            logger.debug("there are nan in image")

        return {'image': image, 'label': label}


class CSV_PNG_Dataset_2D(BaseDataset):
    def __init__(self,file_list="./dataset/train.txt",
                image_paras = {'width':100,'height':256,'channel':1},
                label_paras ={'width':256,'height':10,'channel':3},
                color_space = "Lab",
                is_label_normalized=True):
        super(CSV_PNG_Dataset_2D,self).__init__(file_list, image_paras, label_paras)
        self.color_space = color_space
        self.is_label_normalized = is_label_normalized

    def __getitem__(self, idx):
        image_file_path = self.image_paths[idx]
        label_file_path = self.label_paths[idx]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('{}\t{}'.format(image_file_path, label_file_path))

        image = np.genfromtxt(image_file_path, delimiter=',')
        label = cv2.imread(label_file_path)

        # convert label to a specific color space
        if (self.color_space == "Lab"):
            label = cv2.cvtColor(label, cv2.COLOR_BGR2Lab)
        elif (self.color_space == "Hsl"):
            label = cv2.cvtColor(label, cv2.COLOR_BGR2HSV)

        image = image.reshape(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNEL)
        label = label.reshape(self.LABEL_HEIGHT, self.LABEL_WIDTH, self.LABEL_CHANNEL)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        if self.color_space == "Lab":
            image[int(self.IMAGE_HEIGHT / 4)][0] = 0.0
            image[int(self.IMAGE_HEIGHT * 3 / 4)][0] = 0.0

            image[int(self.IMAGE_HEIGHT / 4)][self.IMAGE_WIDTH - 1] = 0.0
            image[int(self.IMAGE_HEIGHT * 3 / 4)][self.IMAGE_WIDTH - 1] = 0.0



        maxV = torch.max(image)
        minV = torch.min(image)
        diffV = (maxV - minV) * 1.0
        image = (image - minV) / diffV

        image = image.permute(2,0,1)
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

    dataset = CSV_PNG_Dataset(file_list="./dataset/train.txt")
    dataloader = DataLoader(dataset, batch_size=1)

    for ind, batch in enumerate(dataloader):
        if (ind == 0):
            image = batch['image']
            label = batch['label']
            # statistic info
            # loga(image)
            # loga(label)
            label = label.numpy()
            label = np.transpose(label[0], (1,2,0)) * 255.
            label = label.astype(np.uint8)
            label = cv2.cvtColor(label, cv2.COLOR_Lab2RGB)
            plt.figure()
            plt.imshow(label)
            plt.show()
        else:
            break




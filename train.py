import argparse
import os
import time

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import CSV_PNG_Dataset, CSV_PNG_Dataset_2D, PNG_PNG_Dataset
from netArchitecture.VGG import VGGModel, VGGModel_2D
from netArchitecture.ResNet import ResNet18_2D
from visualize import Visualizations

import logging

logger = logging.getLogger("In train.py")
logger.setLevel(logging.DEBUG)
logger.disabled = True

#parse parameters
parser = argparse.ArgumentParser(description='train deep color extraction model')
parser.add_argument('--mode', default=0, type=int)
parser.add_argument('--backbone', default="vgg", type=str)
parser.add_argument('--net_name', default="VGG+ASPP+2D", type=str)
parser.add_argument('--trained_model_config', default="", type=str)  # used for resuming to train network
parser.add_argument('--cuda_device', default=3, type=int)

parser.add_argument('--with_aspp', default=True, choices=('True','False'))
parser.add_argument('--legend_width', default=256, type=int)
parser.add_argument('--legend_height', default=10, type=int)

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--train_bs',default=8, type=int)
parser.add_argument('--num_epochs', default=15, type=int)
parser.add_argument('--color_space', default="Lab", type=str)  # possible value: "Lab", "Rgb"
parser.add_argument('--is_label_normalized', default=True, choices=('True','False'))
parser.add_argument('--loss_function', default="MSE", type=str)

parser.add_argument('--prefix', default="", type=str)

opt = parser.parse_args()
IS_LABEL_NORMALIZED = opt.is_label_normalized == 'True'
WITH_ASPP = opt.with_aspp == 'True'

LEARNING_RATE = opt.lr
BATCH_SIZE = opt.train_bs
NUM_EPOCHS = opt.num_epochs
NET_NAME = opt.net_name
CUDA_DEVICE = opt.cuda_device

MODE = opt.mode
BACKBONE = opt.backbone
COLOR_SPACE = opt.color_space
TRAINED_MODEL_CONFIG = opt.trained_model_config
TRAINED_EPOCH = 0

LOSS_FUNCTION = opt.loss_function

PREFIX = opt.prefix # used in inference.py

IS_NOT_DEBUG = True
USE_VISDOM = True

# if (MODE == 0):   # lab 3D histogram
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNEL = 32
if (MODE == 1): # lab 2D histogram
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    IMAGE_CHANNEL = 1
elif (MODE == 2):   # lab original images
    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 128
    IMAGE_CHANNEL = 3

LABEL_WIDTH = opt.legend_width
LABEL_HEIGHT = opt.legend_height
LABEL_CHANNEL = 3


config = "Net_{}__mode_{}__backbone_{}_colorspace_{}__labelnormalized_{}__lossfun_{}__woaspp_{}__lheight_{}__bs_{}__ep_{}__lr_{}".\
    format(NET_NAME, MODE, BACKBONE, COLOR_SPACE, IS_LABEL_NORMALIZED, LOSS_FUNCTION, WITH_ASPP, LABEL_HEIGHT, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE ) \
    if (TRAINED_MODEL_CONFIG == "") else TRAINED_MODEL_CONFIG


# path for save and load netArchitecture
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, config)

torch.cuda.set_device(CUDA_DEVICE)


# define dataset
# if MODE == 0:
train_set = CSV_PNG_Dataset(
    label_paras ={'width':LABEL_WIDTH,'height':LABEL_HEIGHT,'channel':LABEL_CHANNEL},
    image_paras={'width':IMAGE_WIDTH, 'height':IMAGE_HEIGHT, 'channel':IMAGE_CHANNEL},
    is_label_normalized= IS_LABEL_NORMALIZED
    )
eval_set = CSV_PNG_Dataset(
    label_paras ={'width':LABEL_WIDTH,'height':LABEL_HEIGHT,'channel':LABEL_CHANNEL},
    image_paras={'width':IMAGE_WIDTH, 'height':IMAGE_HEIGHT, 'channel':IMAGE_CHANNEL},
    file_list="./dataset/evaluation.txt",
    is_label_normalized= IS_LABEL_NORMALIZED
    )

if MODE == 1:
    train_set = CSV_PNG_Dataset_2D(
        image_paras={'width':IMAGE_WIDTH,'height':IMAGE_HEIGHT,'channel':IMAGE_CHANNEL},
        label_paras={'width': LABEL_WIDTH, 'height': LABEL_HEIGHT, 'channel': LABEL_CHANNEL},
        color_space=COLOR_SPACE,
        is_label_normalized=IS_LABEL_NORMALIZED)
    eval_set = CSV_PNG_Dataset_2D(
        file_list="./dataset/evaluation.txt",   # here change to evaluation.txt
        image_paras={'width': IMAGE_WIDTH, 'height': IMAGE_HEIGHT, 'channel': IMAGE_CHANNEL},
        label_paras={'width': LABEL_WIDTH, 'height': LABEL_HEIGHT, 'channel': LABEL_CHANNEL},
        color_space=COLOR_SPACE,
        is_label_normalized=IS_LABEL_NORMALIZED)
elif MODE == 2:
    train_set = PNG_PNG_Dataset(label_paras ={'width':LABEL_WIDTH,'height':LABEL_HEIGHT,'channel':LABEL_CHANNEL},
                                color_space=COLOR_SPACE, is_label_normalized=IS_LABEL_NORMALIZED)
    eval_set = PNG_PNG_Dataset(label_paras ={'width':LABEL_WIDTH,'height':LABEL_HEIGHT,'channel':LABEL_CHANNEL},file_list="./dataset/evaluation.txt",
                               color_space=COLOR_SPACE, is_label_normalized=IS_LABEL_NORMALIZED)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=IS_NOT_DEBUG, num_workers=2, drop_last=True)
eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False)

# define net, criterion and optimizer
net = VGGModel(input_channel=IMAGE_CHANNEL, label_height=LABEL_HEIGHT, label_width=LABEL_WIDTH)
if MODE == 2 or MODE == 1:
    if BACKBONE == "vgg":
        net = VGGModel_2D(input_channel=IMAGE_CHANNEL, label_height=LABEL_HEIGHT, label_width=LABEL_WIDTH, with_aspp=WITH_ASPP)
    elif BACKBONE == "resnet18":
        print("resnet18")
        net = ResNet18_2D(input_channel=IMAGE_CHANNEL, label_height=LABEL_HEIGHT, label_width=LABEL_WIDTH, with_aspp=WITH_ASPP)

test_loss_for_each_epoch = [] # used for recording avg mean of each epoch in testing phrase
loss_for_each_epoch = []  # used for recording avg mean of each epoch in training phrase
time_used_cumulation = []
if TRAINED_MODEL_CONFIG != "":
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    TRAINED_EPOCH = checkpoint['epoch'] + 1
    time_used_cumulation = checkpoint['time_used']
    loss_for_each_epoch = checkpoint['loss_for_each_epoch']

print('#netArchitecture parameters:', sum(param.numel() for param in net.parameters()))

if torch.cuda.is_available():
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(torch.cuda.current_device())

    ts = time.time()
    net.cuda()
    print("finished loading netArchitecture params to cuda, time elapsed: {}".format(time.time() - ts))

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

vis = Visualizations(env=config)

if LOSS_FUNCTION == "MSE":
    criterian = nn.MSELoss()
elif LOSS_FUNCTION == "BCE":
    criterian = nn.BCELoss()

sigmoid = torch.nn.Sigmoid()

def train():
    if len(time_used_cumulation) == 0:
        time_used = 0.0
    else:
        time_used = time_used_cumulation[-1]

    iter_100_loss_values = []
    for epoch in range(NUM_EPOCHS - TRAINED_EPOCH):
        tm_start_each_epoch = time.time()
        true_epoch = epoch + TRAINED_EPOCH
        net.train()

        loss_values = []  # used for visdom to visualize
        epoch_loss_value_in_one_epoch = [] # used for recording all loss values in an epoch and computing the mean of them

        for iter, batch in enumerate(train_loader):
            torch.autograd.set_detect_anomaly(True)

            images, labels = batch['image'], batch['label']

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()


            preds = net(images)


            if IS_LABEL_NORMALIZED:
                preds = sigmoid(preds)
            if LOSS_FUNCTION == "MSE":
                loss = criterian(labels, preds)
            elif LOSS_FUNCTION == "BCE":
                loss = criterian(preds, labels.detach())

            loss_values.append(loss.item())
            epoch_loss_value_in_one_epoch.append(loss.item())
            optimizer.zero_grad()

            with torch.autograd.detect_anomaly():
                loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(true_epoch, iter, loss.item()))
                niter = true_epoch * len(train_loader) + iter
                if niter % 100 == 0:
                    iter_100_loss_values.append(np.mean(loss_values))
                if USE_VISDOM:
                    vis.plot_loss(np.mean(loss_values), niter)
                    vis.plot_ground_truth(labels, COLOR_SPACE, caption="groud_truth_in epoch{}, iter{}".format(true_epoch, iter))
                    vis.plot_test_pred(preds, COLOR_SPACE, caption="pred_in epoch{}, iter{}".format(true_epoch, iter))
                    if MODE == 2:
                        vis.plot_ground_truth(images,COLOR_SPACE, win="original images", caption="image in epoch{}, iter{}".format(true_epoch, iter))
                loss_values.clear()
                vis.save()

        time_used = time_used + time.time() - tm_start_each_epoch
        time_used_cumulation.append(time_used)
        loss_for_each_epoch.append(np.mean(epoch_loss_value_in_one_epoch))
        epoch_loss_value_in_one_epoch.clear()
        torch.save({
            'epoch': true_epoch,
            'model_state_dict': net.state_dict(),
            'time_used': time_used_cumulation,
            'loss_for_each_epoch':loss_for_each_epoch
        }, model_path)


def eval(epoch):
    net.eval()
    loss_value_in_epoch = []
    for iter, batch in enumerate(eval_loader):
        images, labels = batch['image'], batch['label']
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        preds = net(images)
        if IS_LABEL_NORMALIZED:
            preds = sigmoid(preds)
        if LOSS_FUNCTION == "MSE":
            loss = criterian(labels, preds)
        elif LOSS_FUNCTION == "BCE":
            loss = criterian(preds, labels.detach())

        loss_value_in_epoch.append(loss.item())
        if USE_VISDOM:
            vis.plot_ground_truth(labels, COLOR_SPACE, win="evaluate_ground_truth")
            vis.plot_test_pred(preds, COLOR_SPACE, win="evaluate_test_pred")
    test_loss_for_each_epoch.append(np.mean(loss_value_in_epoch))

if __name__=="__main__":
    eval(0)
    train()
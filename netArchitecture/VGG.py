import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from .ASPP import aspp

class VGGModel(nn.Module):
    def __init__(self, input_channel=64, label_height=10, label_width=256):
        super(VGGModel, self).__init__()
        if input_channel == 32:
            self.backbone = nn.Sequential(
                nn.Conv2d(input_channel, 64, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),

                nn.Conv2d(128, 256, kernel_size=3, stride=2,padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
                nn.ReLU(),

                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )

        self.aspp = aspp(depth=512)
        self.fc = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 3, kernel_size=1, stride=1)
        )

        self.label_height = label_height
        self.label_width = label_width

        self._init_weight()

        return

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.fc(x)

        legend_pred_resize = nn.functional.interpolate(x, size=(self.label_height, self.label_width),mode='bilinear')

        return legend_pred_resize

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class VGGModel_2D(nn.Module):
    def __init__(self, input_channel=1, label_height=10, label_width=256, with_aspp=True):
        super(VGGModel_2D, self).__init__()
        self.with_aspp = with_aspp
        self.cnv1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU())

        self.cnv1b = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=7, stride=1, padding=3),
            nn.ReLU())

        self.cnv2 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=5, stride=2, padding=2),
            nn.ReLU())

        self.cnv2b = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=5, stride=1, padding=2),
            nn.ReLU())

        self.cnv3 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1),
            nn.ReLU())

        self.cnv3b = nn.Sequential(
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.cnv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU())

        self.cnv4b = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.cnv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU())

        self.cnv5b = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.aspp = aspp(depth=512)

        self.cnv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.cnv6b = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.legend_pred = nn.Conv2d(512, 3, kernel_size=1, stride=1, padding=0)

        self.label_height = label_height
        self.label_width = label_width

        self._init_weight()

        return

    def forward(self, x):
        if(self.with_aspp):
            x_cnv1 = self.cnv1(x)
            x_cnv1b = self.cnv1b(x_cnv1)
            x_cnv2 = self.cnv2(x_cnv1b)
            x_cnv2b = self.cnv2b(x_cnv2)
            x_cnv3 = self.cnv3(x_cnv2b)
            x_cnv3b = self.cnv3b(x_cnv3)
            x_cnv4 = self.cnv4(x_cnv3b)
            x_cnv4b = self.cnv4b(x_cnv4)
            x_cnv5 = self.cnv5(x_cnv4b)
            x_cnv5b = self.cnv5b(x_cnv5)
            x_aspp = self.aspp(x_cnv5b)
            x_cnv6 = self.cnv6(x_aspp)
            x_cnv6b = self.cnv6b(x_cnv6)
            x = self.legend_pred(x_cnv6b)
        else:
            x_cnv1 = self.cnv1(x)
            x_cnv1b = self.cnv1b(x_cnv1)
            x_cnv2 = self.cnv2(x_cnv1b)
            x_cnv2b = self.cnv2b(x_cnv2)
            x_cnv3 = self.cnv3(x_cnv2b)
            x_cnv3b = self.cnv3b(x_cnv3)
            x_cnv4 = self.cnv4(x_cnv3b)
            x_cnv4b = self.cnv4b(x_cnv4)
            x_cnv5 = self.cnv5(x_cnv4b)
            x_cnv5b = self.cnv5b(x_cnv5)
            x_cnv6 = self.cnv6(x_cnv5b)
            x_cnv6b = self.cnv6b(x_cnv6)
            x = self.legend_pred(x_cnv6b)

        legend_pred_resize = nn.functional.interpolate(x, size=(self.label_height, self.label_width), mode='bilinear')

        return legend_pred_resize

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ResNet_2D(nn.Module):
    def __init__(self, input_channel=1, label_height=10, label_width=256, with_aspp=True):
        super(ResNet_2D, self).__init__()
        self.with_aspp = with_aspp
        self.resnet = models.resnet18(pretrained=False)
        self.num_ftrs = self.resnet.fc.in_features
        self.aspp = aspp(depth=self.num_ftrs)
        self.legend_pred = nn.Conv2d(self.num_ftrs, 3, kernel_size=1, stride=1, padding=0)

        self.label_height = label_height
        self.label_width = label_width

    def forward(self, x):
        if(self.with_aspp):
            self.resnet.fc = self.aspp
            x = self.resnet(x)
            x = self.legend_pred(x)
        else:
            self.resnet.fc = nn.Linear(self.num_ftrs, 2)
            x = self.resnet(x)

        legend_pred_resize = nn.functional.interpolate(x, size=(self.label_height, self.label_width), mode='bilinear')
        return legend_pred_resize






if __name__=="__main__":
    from torchsummary import summary
    model = ResNet_2D(with_aspp=False)
    torch.cuda.set_device(3)
    model.cuda()
    summary(model, (3, 256, 128))

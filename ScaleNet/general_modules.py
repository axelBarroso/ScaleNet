import torchvision.models as models
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import cv2

eps_fea_norm = 1e-5
eps_sqrt_dis = 1e-6
eps_l2_norm = 1e-10


def get_scale(model_scale, img1_path, img2_path, image_transforms, device):
    """
    It computes the scale factor between images A and B
    Inputs:
        - model_scale: ScaleNet model
        - img1_path: path to image A
        - img2_path: path to image B
        - image_transforms: image transformations before ScaleNet model
        - device: cpu/gpu device
    Outputs:
        - scale: Scale factor
    """
    img1_ = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    img2_ = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

    sr_H, sr_W, _ = img1_.shape
    tg_H, tg_W, _ = img2_.shape
    crop_sr = min(sr_H, sr_W)
    crop_tg = min(tg_H, tg_W)
    crop = min(crop_tg, crop_sr)
    img1_crop, _, _ = center_crop(img1_, crop)
    img2_crop, _, _ = center_crop(img2_, crop)
    img1_crop = cv2.resize(img1_crop, (240, 240))
    img2_crop = cv2.resize(img2_crop, (240, 240))

    img1 = image_transforms(img1_crop).unsqueeze(0).to(device)
    img2 = image_transforms(img2_crop).unsqueeze(0).to(device)

    scale = model_scale(img1, img2)
    return scale.detach().to('cpu').numpy()[0]


def compute_scale_rep(scale_levels, factor_base=np.sqrt(2)):
    """
    It computes the scale classes in the distribution representation
    Inputs:
        - scale_levels: number of scale classes
        - factor_base: scale factor base for the scale classes
    Outputs:
        - scale_base: scale classes
    """

    base = np.log(factor_base)
    sub_levels = scale_levels // 2
    scale_base = []
    for i in range(scale_levels):
        scale_base.append(base * (i - sub_levels))

    scale_base = np.reshape(scale_base, (1, scale_levels))
    return scale_base


def center_crop(img, size):
    """
    Get the center crop of the input image
    Args:
        img: input image [BxCxHxW]
        size: size of the center crop (tuple)
    Output:
        img_pad: center crop
        x, y: coordinates of the crop
    """

    if not isinstance(size, tuple):
        size = (size, size)

    img = img.copy()
    w, h = img.shape[1::-1]

    pad_w = 0
    pad_h = 0
    if w < size[0]:
        pad_w = np.uint16((size[0] - w) / 2)
    if h < size[1]:
        pad_h = np.uint16((size[1] - h) / 2)
    img_pad = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    w, h = img_pad.shape[1::-1]

    x1 = w // 2 - size[0] // 2
    y1 = h // 2 - size[1] // 2

    img_pad = img_pad[y1:y1 + size[1], x1:x1 + size[0], :]

    return img_pad, x1, y1


class CorrelationVolume(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self):
        super(CorrelationVolume, self).__init__()

    def forward(self, feature_A, feature_B):

        b, c, h, w = feature_A.size()

        # reshape features for matrix multiplication
        feature_A = feature_A.view(b, c, h * w).transpose(1, 2)
        feature_B = feature_B.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_mul = torch.bmm(feature_A, feature_B)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self, axis=1):
        super(FeatureL2Norm, self).__init__()
        self.axis = axis
    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), self.axis) + epsilon, 0.5).unsqueeze(self.axis).expand_as(feature)
        return torch.div(feature, norm)


class ResNetPyramid(nn.Module):
    """
    ResNet model architecture
    """
    def __init__(self, train=False):
        super().__init__()
        self.n_levels = 5
        self.model = models.resnet101(pretrained=True)
        modules = OrderedDict()

        self.resnet_module_list = [self.model.conv1,
                              self.model.bn1,
                              self.model.relu,
                              self.model.maxpool,
                              self.model.layer1,
                              self.model.layer2,
                              self.model.layer3,
                              self.model.layer4]

        modules['level_0'] = nn.Sequential(*[self.model.conv1,
                                              self.model.bn1,
                                              self.model.relu]) #H_2
        for param in modules['level_0'].parameters():
            param.requires_grad = train

        modules['level_1'] = nn.Sequential(*[self.model.maxpool,
                                             self.model.layer1])  # H_4
        for param in modules['level_1'].parameters():
            param.requires_grad = train

        modules['level_2'] = nn.Sequential(*[self.model.layer2]) #H/8
        for param in modules['level_2'].parameters():
            param.requires_grad = train
        modules['level_3'] = nn.Sequential(*[self.model.layer3]) #H/16
        for param in modules['level_3'].parameters():
            param.requires_grad = train
        self.__dict__['_modules'] = modules

    def forward(self, x, quarter_resolution_only=False, eigth_resolution=False):
        outputs = []
        if quarter_resolution_only:
            x_half = self.__dict__['_modules']['level_0'](x)
            x_quarter = self.__dict__['_modules']['level_1'](x_half)
            outputs.append(x_quarter)
        elif eigth_resolution:
            x_half = self.__dict__['_modules']['level_0'](x)
            x_quarter = self.__dict__['_modules']['level_1'](x_half)
            outputs.append(x_quarter)
            x_eight = self.__dict__['_modules']['level_2'](x_quarter)
            outputs.append(x_eight)
        else:
            for layer_n in range(0, self.n_levels-1):
                x = self.__dict__['_modules']['level_' + str(layer_n)](x)
                outputs.append(x)
        return outputs


class VGGPyramid(nn.Module):
    """
        VGG model architecture
    """
    def __init__(self, pretrained=True, fine_tune=False):
        super().__init__()
        self.n_levels = 5
        source_model = models.vgg16(pretrained=pretrained)

        modules = OrderedDict()
        tmp = []
        n_block = 0
        first_relu = False

        for c in source_model.features.children():
            if (isinstance(c, nn.ReLU) and not first_relu) or (isinstance(c, nn.MaxPool2d)):
                first_relu = True
                tmp.append(c)
                modules['level_' + str(n_block)] = nn.Sequential(*tmp)

                if pretrained and not fine_tune:
                    for param in modules['level_' + str(n_block)].parameters():
                        param.requires_grad = False

                tmp = []
                n_block += 1
            else:
                tmp.append(c)

            if n_block == self.n_levels:
                break

        self.__dict__['_modules'] = modules

    def forward(self, x):
        outputs = []
        for layer_n in range(0, self.n_levels):
            x = self.__dict__['_modules']['level_' + str(layer_n)](x)
            outputs.append(x)

        return outputs


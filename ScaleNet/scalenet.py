from ScaleNet.general_modules import VGGPyramid, ResNetPyramid, compute_scale_rep
from ScaleNet.modules import features_to_distribution
import torch.nn as nn
import torch

class scalenet_network(nn.Module):
    '''
        Defines ScaleNet architecture
    '''
    def __init__(self, extractor='VGG', is_test=True, device='cpu', scale_levels=13,
                 add_corr_a=True, add_corr_b=True, multi_scale=True):
        super(scalenet_network, self).__init__()

        self.softmax = torch.nn.Softmax(dim=1).to(device)
        self.is_test = is_test
        scale_base = compute_scale_rep(scale_levels)
        self.scale_base = torch.from_numpy(scale_base).float().to(device)

        fine_tune_extractor = False
        if extractor == 'VGG':
            self.extractor = VGGPyramid(pretrained=True, fine_tune=fine_tune_extractor)
            channel_dim = 512
        else:
            self.extractor = ResNetPyramid(fine_tune_extractor)
            channel_dim = 1024

        size_map = 15
        in_channels = size_map * size_map

        self.to_scale = features_to_distribution(in_channels=in_channels, out_channels=scale_levels,
                                                 channel_dim=channel_dim, device=device, add_corr_a=add_corr_a,
                                                 add_corr_b=add_corr_b, multi_scale=multi_scale)


    def transform_distr_to_factor(self, scale_distr):

        scale_distr = self.softmax(scale_distr)

        scale_factor = torch.exp(torch.sum(scale_distr * self.scale_base, dim=1))

        return scale_factor

    def forward(self, x1, x2):
        """
        x1 - target image
        x2 - source image
        """

        src_feats = self.extractor(x1)[-1]
        tgt_feats = self.extractor(x2)[-1]

        scale_a2b = self.to_scale(src_feats, tgt_feats)

        if self.is_test:
            scale_b2a = self.to_scale(tgt_feats, src_feats)

            scale_factor = self.transform_distr_to_factor(scale_a2b)
            scale_factor_inv = self.transform_distr_to_factor(scale_b2a)
            scale_factor = torch.exp((torch.log(scale_factor) - torch.log(scale_factor_inv)) / 2.)

            return scale_factor

        return scale_a2b
from ScaleNet.general_modules import CorrelationVolume, FeatureL2Norm
import torch
import torch.nn as nn


class ASPP_module(nn.Module):
    """
    ASPP_module: Extracts multi-scale features and fuses them in a single feature map
    """

    def __init__(self, ms_levels, channel_dim):
        super(ASPP_module, self).__init__()

        self.ms_levels = ms_levels
        self.ms_kernels = []
        learn_ms = {}
        for idx_ms in range(self.ms_levels):

            if idx_ms == 0:
                learn_ms['ms_{}'.format(idx_ms)] = nn.Sequential(nn.Conv2d(channel_dim, channel_dim, (1, 1)),
                                                                 nn.BatchNorm2d(channel_dim), nn.LeakyReLU(0.1))
            else:
                learn_ms['ms_{}'.format(idx_ms)] = nn.Sequential(
                    nn.Conv2d(channel_dim, channel_dim, (3, 3), padding=idx_ms,
                              dilation=idx_ms), nn.BatchNorm2d(channel_dim), nn.LeakyReLU(0.1))

        self.__dict__['_modules'] = learn_ms
        self.mix_comb = nn.Sequential(
            nn.Conv2d(channel_dim * self.ms_levels, channel_dim * self.ms_levels // 2, (1, 1), padding=0),
            nn.BatchNorm2d(channel_dim * self.ms_levels // 2), nn.LeakyReLU(0.1),
            nn.Conv2d(channel_dim * self.ms_levels // 2, channel_dim, (1, 1), padding=0))

    def forward(self, src_feats, tgt_feats):

        for idx_ms in range(self.ms_levels):

            tmp_filter = self.__dict__['_modules']['ms_{}'.format(idx_ms)]
            output_src = tmp_filter(src_feats)
            output_trg = tmp_filter(tgt_feats)

            if idx_ms == 0:
                ms_src = output_src
                ms_trg = output_trg
            else:
                ms_src = torch.cat([ms_src, output_src], dim=1)
                ms_trg = torch.cat([ms_trg, output_trg], dim=1)

        src_feats = self.mix_comb(ms_src)
        tgt_feats = self.mix_comb(ms_trg)

        return src_feats, tgt_feats


class features_to_distribution(nn.Module):
    """
    features_to_distribution: Computes the scale distribution from extracted features
    """
    def __init__(self, in_channels, out_channels=1, channel_dim=512,
                 add_corr_a=False, add_corr_b=False, multi_scale=True, device='gpu'):
        super(features_to_distribution, self).__init__()

        self.device = device
        self.out_channels = out_channels
        self.corr = CorrelationVolume().to(self.device)
        self.l2norm = FeatureL2Norm().to(self.device)
        self.relu = nn.ReLU()

        self.add_corr_a = add_corr_a
        self.add_corr_b = add_corr_b

        self.reduce_vol_dim = reduce_vol_block(in_channels)
        if add_corr_a:
            self.reduce_vol_a_dim = reduce_vol_block(in_channels)
        if add_corr_b:
            self.reduce_vol_b_dim = reduce_vol_block(in_channels)

        self.multi_scale = multi_scale
        if multi_scale:
            ms_levels = 4
            self.ms_extractor = ASPP_module(ms_levels, channel_dim)

        n_blocks = 1
        n_blocks += 1 if add_corr_a else 0
        n_blocks += 1 if add_corr_b else 0
        output_corr = 1568

        self.to_distr = compute_distribution(out_channels, output_corr, n_blocks)

    def forward(self, featsA, featsB):

        if self.multi_scale:
            featsA, featsB = self.ms_extractor(featsA, featsB)

        featsA_norm = self.l2norm(featsA)
        featsB_norm = self.l2norm(featsB)

        volume_corr_A_to_B = self.l2norm(self.relu(self.corr(featsA_norm, featsB_norm)))

        feats_A_to_B = self.reduce_vol_dim(volume_corr_A_to_B)

        if self.add_corr_a:
            volume_corrA = self.l2norm(self.relu(self.corr(featsA_norm, featsA_norm)))
            out_corrA = self.reduce_vol_a_dim(volume_corrA)
            feats_A_to_B = torch.cat([feats_A_to_B, out_corrA], dim=1)

        if self.add_corr_b:
            volume_corrB = self.l2norm(self.relu(self.corr(featsB_norm, featsB_norm)))
            out_corrB = self.reduce_vol_b_dim(volume_corrB)
            feats_A_to_B = torch.cat([feats_A_to_B, out_corrB], dim=1)

        feats_A_to_B = feats_A_to_B.unsqueeze(-1).unsqueeze(-1)
        scale_dist = self.to_distr(feats_A_to_B)

        b, _, _, _ = scale_dist.size()
        scale_dist = scale_dist.reshape((b, self.out_channels))

        return scale_dist


def reduce_vol_block(in_channels):
    """
        CNN block to reduce the correlation volume dimensionality
    """
    return nn.Sequential(nn.Conv2d(in_channels, 256, (3, 3)), nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
                         nn.Conv2d(256, 128, (3, 3)), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
                         nn.Conv2d(128, 64, (3, 3)), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
                         nn.Conv2d(64, 32, (3, 3)), nn.BatchNorm2d(32), nn.LeakyReLU(0.1), nn.Flatten())


def compute_distribution(out_channels, input_size, n_blocks, drop_rate = 0.3):
    """
        1D CNN to map the correlation volumes into the scale factor distribution
    """
    return nn.Sequential(nn.Conv2d(input_size * n_blocks, 2056, (1, 1)), nn.BatchNorm2d(2056), nn.LeakyReLU(0.1),
                        nn.Conv2d(2056, 1024, (1, 1)), nn.BatchNorm2d(1024), nn.LeakyReLU(0.1),
                        nn.Conv2d(1024, 512, (1, 1)), nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
                        nn.Dropout(drop_rate),
                        nn.Conv2d(512, 256, (1, 1)), nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
                        nn.Dropout(drop_rate),
                        nn.Conv2d(256, 128, (1, 1)), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
                        nn.Dropout(drop_rate),
                        nn.Conv2d(128, out_channels, (1, 1)))

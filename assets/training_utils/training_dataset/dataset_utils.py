import torch

class AddGaussianNoise(object):
    '''
        Class to add gaussian additive noise
    '''
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def read_dataset_pairs(txt_path, max_scale=10.):
    '''
    Function read the dataset pairs
        :param txt_path: path to the dataset pairs
        :param max_scale: optional parameter to control the maximum scale value
        :return: dataset pairs
    '''

    file1 = open(txt_path, 'r')
    pairs = file1.readlines()

    scale_dataset = []
    for idx in range(len(pairs)):
        info_pair = pairs[idx].split('-/-')
        scale_ratio = float(info_pair[7].split('\n')[0])

        if scale_ratio < max_scale and scale_ratio > 1./max_scale:
            new_sample = {'img1': info_pair[5], 'img2': info_pair[6],
                          'scene': info_pair[0], 'scale_ratio': scale_ratio}
            scale_dataset.append(new_sample)

    return scale_dataset

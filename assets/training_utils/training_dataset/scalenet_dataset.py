import os.path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import PIL
import cv2
from PIL import Image
from os import path as osp
from ScaleNet.general_modules import center_crop
from assets.training_utils.training_dataset.dataset_utils import AddGaussianNoise, read_dataset_pairs

class ScaleNet_dataset(Dataset):
    """
        Dataset class for training/validation ScaleNet architecture.

        Inputs:
            - image_path: indicates the path to the Megadepth dataset images
            - pairs_path: stores the path to the training/validation image pairs
            - output_size: image output size, default: (240, 240)
            - is_debug: indicates if the training is on debug mode (smaller dataset size), default: False
            - is_training: indicates if dataset is loading training or validation pairs
            - scale_levels: indicates the size of the scale distribution, default: 13
            - root_np: indicates path to save tmp files for faster training
    """

    def __init__(self, image_path, pairs_path, output_size=(240, 240), is_debug=False, is_training=True,
                 scale_levels=13, root_np='/assets/data/tmp_data/'):

        super().__init__()

        self.img_path = image_path
        self.pairs_path = pairs_path
        self.scale_levels = scale_levels
        self.scale_base = np.log(np.sqrt(2))
        self.is_training = is_training
        self.use_synthetic_paris = False
        self.H_OUT, self.W_OUT = (output_size)

        # color augmentation
        self.brightness = (0.6, 1.4)
        self.contrast = (0.6, 1.4)
        self.saturation = (0.6, 1.4)
        self.hue = (-0.4, 0.4)

        # Gaussian noise and black areas augmentation
        self.gaussian_noise = AddGaussianNoise(0.1, 0.08)
        self.random_erasing = transforms.RandomErasing()

        # normalise and transform
        mean_vector = np.array([0.485, 0.456, 0.406])
        std_vector = np.array([0.229, 0.224, 0.225])
        self.mean = torch.as_tensor(mean_vector, dtype=torch.float32)
        self.std = torch.as_tensor(std_vector, dtype=torch.float32)
        self.to_tensor = transforms.ToTensor()
        self.to_PIL = transforms.ToPILImage()

        # it creates a tmp folder to make faster the training of ScaleNet with preprocessed images
        self.root_np = root_np
        if not osp.isdir(self.root_np):
            os.mkdir(self.root_np)

        self.scale_distr = self.build_scale_rep()
        self.max_scale_training = 7.

        # load dataset pairs
        df = read_dataset_pairs(pairs_path)

        self.df = np.asarray(df)[np.random.permutation(range(len(df)))]
        self.indices = np.random.permutation(range(len(df)))

        if is_debug:
            if is_training:
                num_im_debug = 50000
            else:
                num_im_debug = 5000

            self.df = self.df[:num_im_debug]

    def build_scale_rep(self):

        sub_levels = self.scale_levels // 2
        scale_distr = []

        for i in range(self.scale_levels):
            scale_distr.append(self.scale_base * (i - sub_levels))

        return np.asarray(scale_distr)

    def scale_factor_to_distribution(self, scale):

        b = self.scale_distr

        ln_scale = np.log(scale)
        label_prob = np.zeros((self.scale_levels))
        i_scale = min(max(b[0], ln_scale), b[-1])
        idx_sc = np.max(np.where(b <= i_scale))

        if idx_sc == self.scale_levels-1:
            label_prob[idx_sc] = 1.
        else:
            p = (i_scale - b[idx_sc + 1]) / (b[idx_sc] - b[idx_sc + 1])
            label_prob[idx_sc] = p
            label_prob[idx_sc + 1] = 1 - p

        return label_prob

    def get_synthetic_pair(self, img_src):

        augm_prob = 7
        img_size = img_src.shape

        img_dst = self.to_PIL(img_src)

        # Scale augmentation
        scale = np.random.uniform(1.01, self.max_scale_training)
        newsize = (int(scale * img_size[0]), int(scale * img_size[1]))
        scale = (newsize[0] / img_size[0] + newsize[1] / img_size[1]) / 2.
        img_dst = img_dst.resize(newsize)

        # Color augmentation
        color_aug2 = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        img_dst = color_aug2(img_dst)

        # Flip augmentation
        if np.random.randint(11) <= augm_prob:
            img_dst = img_dst.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        # Rotation augmentation
        rot_angle = np.random.uniform(-30.0, 30.0)
        img_dst = img_dst.rotate(rot_angle)

        img_dst = np.array(img_dst)

        return img_dst, scale


    def get_example_from_megadepth(self, data):

        augm_prob = 7
        tmp_dict_file = '{}{}_{}_{}.pkl'.format(self.root_np, data['scene'], data['img1'].split('.jpg')[0],
                                                data['img2'].split('.jpg')[0])

        read_file = True
        if os.path.isfile(tmp_dict_file):
            try:
                a_file = open(tmp_dict_file, "rb")
                example = pickle.load(a_file)
                a_file.close()
            except EOFError:
                read_file = False
        else:
            read_file = False

        # Load synthetic pair
        if self.is_training and self.use_synthetic_paris and np.random.randint(21) < 1:

            if np.random.randint(2):
                path = self.img_path + '/{}/images/{}'.format(data['scene'], data['img1'])
            else:
                path = self.img_path + '/{}/images/{}'.format(data['scene'], data['img2'])

            img_src = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img_target, scale = self.get_synthetic_pair(img_src)

            sr_H, sr_W, _ = img_src.shape
            tg_H, tg_W, _ = img_target.shape
            crop_sr = min(sr_H, sr_W)
            crop_tg = min(tg_H, tg_W)
            crop = min(crop_tg, crop_sr)
            img_src_crop, _, _ = center_crop(img_src, crop)
            img_target_crop, _, _ = center_crop(img_target, crop)

            img_src_crop = cv2.resize(img_src_crop, (self.H_OUT, self.W_OUT))
            img_target_crop = cv2.resize(img_target_crop, (self.H_OUT, self.W_OUT))

        else:
            # Load real pair
            if read_file:

                if self.is_training and np.random.randint(2):
                    img_src_crop = example['img_src_crop']
                    img_target_crop = example['img_target_crop']
                    scale = example['scale_ratio']
                else:
                    img_src_crop = example['img_target_crop']
                    img_target_crop = example['img_src_crop']
                    scale = 1. / example['scale_ratio']

            else:
                path1 = self.img_path + '/{}/images/{}'.format(data['scene'], data['img1'])
                path2 = self.img_path + '/{}/images/{}'.format(data['scene'], data['img2'])

                if self.is_training and np.random.randint(2):
                    im_path1 = path2
                    im_path2 = path1
                    scale = 1. / data['scale_ratio']
                else:
                    im_path1 = path1
                    im_path2 = path2
                    scale = data['scale_ratio']

                img_src = cv2.cvtColor(cv2.imread(im_path1), cv2.COLOR_BGR2RGB)
                img_target = cv2.cvtColor(cv2.imread(im_path2), cv2.COLOR_BGR2RGB)

                sr_H, sr_W, _ = img_src.shape
                tg_H, tg_W, _ = img_target.shape
                crop_sr = min(sr_H, sr_W)
                crop_tg = min(tg_H, tg_W)
                crop = min(crop_tg, crop_sr)
                img_src_crop, _, _ = center_crop(img_src, crop)
                img_target_crop, _, _ = center_crop(img_target, crop)

                img_src_crop = cv2.resize(img_src_crop, (self.H_OUT, self.W_OUT))
                img_target_crop = cv2.resize(img_target_crop, (self.H_OUT, self.W_OUT))

                example = {}
                example['img_src_crop'] = img_src_crop
                example['img_target_crop'] = img_target_crop
                example['scale_ratio'] = scale

                a_file = open(tmp_dict_file, "wb")
                pickle.dump(example, a_file)
                a_file.close()

        if self.is_training and np.random.randint(11) <= augm_prob:
            color_aug1 = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug1 = (lambda x: x)

        if self.is_training and np.random.randint(11) <= augm_prob:
            color_aug2 = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug2 = (lambda x: x)

        cropped_source_image = self.to_PIL(img_src_crop)

        if self.is_training and np.random.randint(11) <= 2 and False:
            cropped_target_image = self.to_PIL(img_src_crop)
            do_scale_aug = False
            scale = 1.
        else:
            cropped_target_image = self.to_PIL(img_target_crop)
            do_scale_aug = True

        # Flip augmentation
        if self.is_training and np.random.randint(11) <= augm_prob:
            cropped_source_image = cropped_source_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if self.is_training and np.random.randint(11) <= augm_prob:
            cropped_target_image = cropped_target_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        # Flip augmentation Up and down
        if self.is_training and np.random.randint(11) <= 2:
            cropped_source_image = cropped_source_image.transpose(PIL.Image.FLIP_TOP_BOTTOM)

        if self.is_training and np.random.randint(11) <= 2:
            cropped_target_image = cropped_target_image.transpose(PIL.Image.FLIP_TOP_BOTTOM)

        # Rotation augmentation
        if self.is_training and np.random.randint(11) <= augm_prob:
            rot_angle = np.random.uniform(-30.0, 30.0)
            cropped_source_image = cropped_source_image.rotate(rot_angle)

        if self.is_training and np.random.randint(11) <= augm_prob:
            rot_angle = np.random.uniform(-30.0, 30.0)
            cropped_target_image = cropped_target_image.rotate(rot_angle)

        # Scale augmentation
        if self.is_training and np.random.randint(11) <= augm_prob and do_scale_aug:
            aug_scale = np.random.uniform(1.01, 1.4)

            tmp_scale = scale / aug_scale
            if tmp_scale > 1./self.max_scale_training:
                newsize = (int(aug_scale * self.H_OUT), int(aug_scale * self.W_OUT))
                aug_scale = (newsize[0] / self.H_OUT + newsize[1] / self.W_OUT) / 2.
                cropped_source_image = cropped_source_image.resize(newsize)
                scale /= aug_scale
                x1 = np.random.randint(0, newsize[0] - self.H_OUT)
                y1 = np.random.randint(0, newsize[1] - self.W_OUT)
                cropped_source_image = cropped_source_image.crop((x1, y1, x1 + self.H_OUT, y1 + self.W_OUT))

        if self.is_training and np.random.randint(11) <= augm_prob and do_scale_aug:
            aug_scale = np.random.uniform(1.01, 1.4)
            tmp_scale = scale * aug_scale
            if tmp_scale < self.max_scale_training:
                newsize = (int(aug_scale * self.H_OUT), int(aug_scale * self.W_OUT))
                aug_scale = (newsize[0] / self.H_OUT + newsize[1] / self.W_OUT) / 2.
                cropped_target_image = cropped_target_image.resize(newsize)
                scale *= aug_scale
                x1 = np.random.randint(0, newsize[0] - self.H_OUT)
                y1 = np.random.randint(0, newsize[1] - self.W_OUT)
                cropped_target_image = cropped_target_image.crop((x1, y1, x1 + self.H_OUT, y1 + self.W_OUT))

        cropped_source_image = color_aug1(cropped_source_image)
        cropped_target_image = color_aug2(cropped_target_image)

        cropped_source_image = self.to_tensor(cropped_source_image).float()
        cropped_target_image = self.to_tensor(cropped_target_image).float()

        # Add Gaussian Noise
        if self.is_training and np.random.randint(11) <= augm_prob:
            cropped_source_image = self.gaussian_noise(cropped_source_image)
        if self.is_training and np.random.randint(11) <= augm_prob:
            cropped_target_image = self.gaussian_noise(cropped_target_image)

        # Add random remove
        if self.is_training and np.random.randint(11) <= 2:
            cropped_source_image = self.random_erasing(cropped_source_image)
        if self.is_training and np.random.randint(11) <= 2:
            cropped_target_image = self.random_erasing(cropped_target_image)

        # Normalise
        cropped_source_image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        cropped_target_image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])

        scale_gt = self.scale_factor_to_distribution(scale)

        return {'source_image': cropped_source_image,
                'target_image': cropped_target_image,
                'scale': scale_gt,
                'scale_factor': scale}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        data = self.df[idx]
        example = self.get_example_from_megadepth(data)

        return example


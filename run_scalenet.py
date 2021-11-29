import argparse

import torch
import torchvision.transforms as transforms
import numpy as np

from ScaleNet.scalenet import scalenet_network as scaleNetwork
from ScaleNet.general_modules import get_scale


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--im1_path', type=str, default='assets/im_test/im1.jpg',
        help='path to first image')

    parser.add_argument(
        '--im2_path', type=str, default='assets/im_test/im2.jpg',
        help='path to second image')

    parser.add_argument(
        '--path_scalenet', type=str, default='ScaleNet/weights/vgg_scalenet_weights.pth',
        help='path to ScaleNet weights')

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    mean_vector = np.array([0.485, 0.456, 0.406])
    std_vector = np.array([0.229, 0.224, 0.225])
    normTransform = transforms.Normalize(mean_vector, std_vector)
    image_transforms = transforms.Compose([transforms.ToTensor(), normTransform])

    # Define ScaleNet model
    model_scale = scaleNetwork(device=device)
    checkpoint = torch.load(args.path_scalenet)

    model_scale.to_scale.load_state_dict(checkpoint['state_dict'])
    model_scale.eval()
    model_scale.to(device)

    scale_factor = get_scale(model_scale, args.im1_path, args.im2_path, image_transforms, device)

    print('Scale factor between image \033[91m{}\033[0m and image \033[91m{}\033[0m is \033[34m{:.2f}\033[0m'
          .format(args.im1_path, args.im2_path, scale_factor))

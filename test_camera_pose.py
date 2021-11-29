from ScaleNet.scalenet import scalenet_network as scaleNetwork
from ScaleNet.general_modules import get_scale
from assets.test_utils.test_utils import parse_F_mat, pose_auc, compute_pose_error, parse_t_mat
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import os
import argparse
import torchvision.transforms as transforms

def test_scalenet_camera_pose():

    # Argument parsing
    parser = argparse.ArgumentParser(description='Scale-Net camera pose test script')

    # Paths
    parser.add_argument('--image_data_path', type=str, default='path-to-Megadepth',
                        help='path to dataset images')

    parser.add_argument('--csv_parse_root', type=str, default='assets/data/test.csv',
                        help='path to the txt files containing image pairs')

    parser.add_argument('--path_scalenet', type=str, default='ScaleNet/weights/vgg_scalenet_weights.pth',
                        help='path to ScaleNet weights')

    parser.add_argument('--max_img', type=int, default=3500,
                        help='maximum image size to extract features')

    parser.add_argument('--min_img', type=int, default=256,
                        help='minimum image size to extract features')

    parser.add_argument('--cuda_device', type=str, default='0', help='Indicates which GPU should be used')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    torch.set_grad_enabled(False)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Define ScaleNet model
    model_scale = scaleNetwork(device=device)
    checkpoint = torch.load(args.path_scalenet)

    model_scale.to_scale.load_state_dict(checkpoint['state_dict'])
    model_scale.eval()
    model_scale.to(device)

    mean_vector = np.array([0.485, 0.456, 0.406])
    std_vector = np.array([0.229, 0.224, 0.225])
    normTransform = transforms.Normalize(mean_vector, std_vector)
    image_transforms = transforms.Compose([transforms.ToTensor(), normTransform])

    # read test set
    test_database = pd.read_csv(args.csv_parse_root)

    # SIFT and FLANN parameters
    sift = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    pose_errors = []
    count_incorrect_pairs = 0
    root = args.image_data_path

    for idx in tqdm(range(len(test_database['scale']))):

        name1 = test_database['img1'][idx]
        name2 = test_database['img2'][idx]
        scene = str(test_database['scene'][idx])
        scene = '0'*(4-len(scene)) + scene

        K1 = parse_F_mat(test_database['K1'][idx])
        K2 = parse_F_mat(test_database['K2'][idx])
        R_gt = parse_F_mat(test_database['R'][idx])
        t_gt = parse_t_mat(test_database['t'][idx])

        path1 = root + '/{}/images/{}'.format(scene, name1)
        path2 = root + '/{}/images/{}'.format(scene, name2)
        scale_factor = get_scale(model_scale, path1, path2, image_transforms, device)

        # Extract features
        img_1 = cv2.imread(path1, 0)
        img_2 = cv2.imread(path2, 0)

        # if scale_factor >= 1.:
        w, h = img_1.shape[1], img_1.shape[0]
        w_new, h_new = int(w * scale_factor), int(h * scale_factor)

        max_value = max(w_new, h_new)

        if max_value > args.max_img:
            w_new = int(w_new * (args.max_img/max_value))
            h_new = int(h_new * (args.max_img/max_value))
        elif max_value < args.min_img:
            w_new = int(w_new * (args.min_img / max_value))
            h_new = int(h_new * (args.min_img / max_value))

        scale_factor_tmp = (float(w) / float(w_new), float(h) / float(h_new))
        img_1 = cv2.resize(img_1, (w_new, h_new))

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img_1, None)
        kp2, des2 = sift.detectAndCompute(img_2, None)

        matches = flann.knnMatch(des1, des2, k=2)
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for j, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                pts2.append(np.asarray(kp2[m.trainIdx].pt))
                pts1.append(np.asarray(kp1[m.queryIdx].pt) * scale_factor_tmp)

        # Compute the pose
        f_mean = np.mean([K1[0, 0], K1[1, 1], K2[0, 0], K2[1, 1]])
        norm_thresh = 1. / f_mean

        if len(pts1) < 6:
            count_incorrect_pairs += 1
            pose_errors.append(180)
        else:
            src_pts_norm = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
            dst_pts_norm = (pts2 - K2[[0, 1], [2, 2]][None]) / K2[[0, 1], [0, 1]][None]

            E, mask = cv2.findEssentialMat(src_pts_norm, dst_pts_norm, np.eye(3),
                                           threshold=norm_thresh, prob=0.99, method=cv2.RANSAC)

            n, R, t, _ = cv2.recoverPose(E, src_pts_norm, dst_pts_norm, np.eye(3), 1e9, mask=mask)
            error_t, error_r = compute_pose_error(R_gt, t_gt, R, t[:, 0])
            error_pose = max(error_t, error_r)
            pose_errors.append(error_pose)

    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.*yy for yy in aucs]
    print('Evaluation results (mean over {} pairs):'.format(len(test_database['scale'])))
    print('AUC@5\t AUC@10\t AUC@20\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs[0], aucs[1], aucs[2]))
    print('Incorrect pairs: ' + str(count_incorrect_pairs))

if __name__ == '__main__':

    test_scalenet_camera_pose()
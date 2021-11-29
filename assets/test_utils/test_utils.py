import numpy as np

def parse_F_mat(F_txt):
    """
    Auxiliary function to parse the fundamental matrix from txt to 3x3 float matrix
    Args:
        F_txt: input txt fundamental matrix
    Output:
        F: Fundamental matrix, Size: (3, 3)
    """
    f_mat = F_txt.split('/')
    f_mat = list(map(lambda x: float(x), f_mat))

    f_mat = np.asarray(f_mat).reshape((3, 3))
    return f_mat

def parse_t_mat(t_txt):
    """
    Auxiliary function to parse the translation vector
    """
    t_mat = t_txt.split('/')
    t_mat = list(map(lambda x: float(x), t_mat))

    t_mat = np.asarray(t_mat).reshape((3))
    return t_mat


def angle_error_mat(R1, R2):
    """
    Code from: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py#L377
    """
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    """
    Code from: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py#L383
    """
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(R_gt, t_gt, R, t):
    """
    Adapted code from: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py#L391
    """
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    """
    Code from: https://github.com/magicleap/SuperGluePretrainedNetwork/
    """
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs
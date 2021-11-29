# ScaleNet: A Shallow Architecture for Scale Estimation

Repository for the code of ScaleNet paper:

```text
"ScaleNet: A Shallow Architecture for Scale Estimation".
Axel Barroso-Laguna, Yurun Tian, and Krystian Mikolajczyk. arxiv 2022.
```
[[Paper on arxiv](https://arxiv.org/abs/  )]

## Prerequisite

Python 3.7 is required for running and training ScaleNet code. Use Conda to install the dependencies:

```bash
conda create --name scalenet_env
conda activate scalenet_env 
conda install pytorch==1.2.0 -c pytorch
conda install -c conda-forge tensorboardx opencv tqdm 
conda install -c anaconda pandas 
conda install -c pytorch torchvision 
```

## Scale estimation

`run_scalenet.py` can be used to estimate the scale factor between two input images. We provide as an example two images, `im1.jpg` and `im2.jpg`, within the _assets/im_test_ folder as an example. For a quick test, please run: 

```bash
python run_scalenet.py --im1_path assets/im_test/im1.jpg --im2_path assets/im_test/im2.jpg
```

Arguments:

  * _im1_path_: Path to image A. 
  * _im2_path_: Path to image B.

It returns the scale factor A->B.

## Training ScaleNet

We provide a list of Megadepth image pairs and scale factors in the _assets_ folder. 
We use the undistorted images, corresponding camera intrinsics, and extrinsics preprocessed by [D2-Net](https://github.com/mihaidusmanu/d2-net). 
You can download them directly from their [main repository](https://github.com/mihaidusmanu/d2-net#downloading-and-preprocessing-the-megadepth-dataset).
If you desire to use the default configuration for training, just run the following line:

```bash
python train_ScaleNet.py --image_data_path /path/to/megadepth_d2net
```

There are though some important arguments to take into account when training ScaleNet. 

Arguments:
    
  * _image_data_path_: Path to the undistorted Megadepth images from D2-Net.
  * _save_processed_im_: ScaleNet processes the images so that they are center-cropped and resized to a default resolution. We give the option to store the processed images and load them during training, which results in a much faster training. However, the size of the files can be big, and hence, we suggest storing them in a large storage disk. Default: True.
  * _root_precomputed_files_: Path to save the processed image pairs.
    
If you desire to modify ScaleNet training or architecture, look for all the arguments in the _train_ScaleNet.py_ script. 

## Test ScaleNet - camera pose 

In addition to the training, we also provide a template for testing ScaleNet in the camera pose task. In _assets/data/test.csv_, you can find the test Megadepth pairs, along with their scale change as well as their camera poses.

Run the following command to test ScaleNet + SIFT in our custom camera pose split: 

```bash
python test_camera_pose.py --image_data_path /path/to/megadepth_d2net
```

_camera_pose.py_ script is intended to provide a structure of our camera pose experiment. You can change either the local feature extractor or the scale estimator and obtain your camera pose results.


## BibTeX

If you use this code or the provided training/testing pairs in your research, please cite our paper:

```bibtex
@InProceedings{Barroso-Laguna2022_scale,
    author = {Barroso-Laguna, Axel and Tian, Yurun and Mikolajczyk, Krystian},
    title = {{ScaleNet: A Shallow Architecture for Scale Estimation}},
    booktitle = {Arxiv: },
    year = {2022},
}

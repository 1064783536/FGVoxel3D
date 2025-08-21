# Installation
Install mmdetection3d and make sure it reproduces the pointpillar algorithm correctly!(https://github.com/open-mmlab/mmdetection3d/tree/main)

# download the code.

# Train
python tools/train.py configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class_multivoxel.py


# Test
python tools/test.py pretrain_model/second_hv_secfpn_8xb6-80e_kitti-3d-3class_multivoxel.py pretrain_model/epoch_40.pth

# Installation
Install mmdetection3d and make sure it reproduces the pointpillar algorithm correctly!(https://github.com/open-mmlab/mmdetection3d/tree/main)

# download the code.

# Train
python tools/train.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py

# Test
python tools/test.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py

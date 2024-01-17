# Camera-Calibration-Reconstruct

this is a repo for stereo depth reconstruction

## 1.directory

```
.
├── config      
├── core        
├── data         
├── demo        
├── libs         
├── scripts       
│   ├── mono_camera_calibration.sh     
│   └── stereo_camera_calibration.sh  
├── get_stereo_images.py                 
├── mono_camera_calibration.py           
├── stereo_camera_calibration.py         
├── requirements.txt                      
└── README.md

```

# 2. Environment

- [requirements.txt](requirements.txt)
- python-pcl
- open3d-python=0.7.0.0
- opencv-python
- opencv-contrib-python

## 3.Calibration

#### (1) left and right

```bash
bash scripts/get_stereo_images.sh
```



|left_image                        |right_image                           |
|:--------------------------------:|:------------------------------------:|
|![Image text](docs/left_chess.png)|![Image text](docs/right_chess.png)   |

#### (2) [monocular](scripts/mono_camera_calibration.sh)

- `bash scripts/mono_camera_calibration.sh`
- if error is over 0,1 redo calibration

```bash
#!/usr/bin/env bash

image_dir=data/lenacv-camera #
save_dir=configs/lenacv-camera #
width=8
height=11
square_size=20 #mm
image_format=png # png,jpg
show=True # 
# left camera calibration
python mono_camera_calibration.py \
    --image_dir  $image_dir \
    --image_format $image_format  \
    --square_size $square_size  \
    --width $width  \
    --height $height  \
    --prefix left  \
    --save_dir $save_dir \
    --show $show

# right camera calibration
python mono_camera_calibration.py \
    --image_dir  $image_dir \
    --image_format  $image_format  \
    --square_size $square_size  \
    --width $width  \
    --height $height  \
    --prefix right  \
    --save_dir $save_dir \
    --show $show
```

under`$save_dir`get`left_cam.yml`and`right_cam.yml`for left and right

#### (3) [stereo](scripts/stereo_camera_calibration.sh)
- `bash scripts/stereo_camera_calibration.sh`
- if reprojection error is over 0.1，redo calibration

```bash
image_dir=data/lenacv-camera # 
save_dir=configs/lenacv-camera # 
width=8
height=11
square_size=20 #mm
image_format=png # png,jpg
#show=True # 
show=False # 
# stereo camera calibration
python stereo_camera_calibration.py \
    --left_file $save_dir/left_cam.yml \
    --right_file $save_dir/right_cam.yml \
    --left_prefix left \
    --right_prefix right \
    --width $width \
    --height $height \
    --left_dir $image_dir \
    --right_dir $image_dir \
    --image_format  $image_format  \
    --square_size $square_size \
    --save_dir $save_dir 
```

use under`$save_dir`there is `stereo_cam.yml`contains the calibration

- 
- 
```
(x,y)=(203,273),depth=633.881653mm
(x,y)=(197,329),depth=640.386047mm
(x,y)=(222,292),depth=631.549072mm
(x,y)=(237,270),depth=630.389221mm
(x,y)=(208,246),depth=652.560669mm
```






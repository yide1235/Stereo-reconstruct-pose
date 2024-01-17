#!/usr/bin/env bash

image_dir_left=data/left/ # 
image_dir_right=data/right/ # 
save_dir=configs/lenacv-camera # 
width=9
height=7
square_size=20 #mm
image_format=jpg # png,jpg
show=True # 
# left camera calibration
python mono_camera_calibration.py \
    --image_dir  $image_dir_left \
    --image_format $image_format  \
    --square_size $square_size  \
    --width $width  \
    --height $height  \
    --prefix left  \
    --save_dir $save_dir \
    --show $show

# right camera calibration
python mono_camera_calibration.py \
    --image_dir  $image_dir_right \
    --image_format  $image_format  \
    --square_size $square_size  \
    --width $width  \
    --height $height  \
    --prefix right  \
    --save_dir $save_dir \
    --show $show

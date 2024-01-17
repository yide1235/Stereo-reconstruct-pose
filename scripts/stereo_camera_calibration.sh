#!/usr/bin/env bash

image_dir_left=data/left/ # 棋盘格图片
image_dir_right=data/right/ # 棋盘格图片
save_dir=configs/lenacv-camera # 保存标定结果
width=9
height=7
square_size=20 #mm
image_format=jpg # 图片格式，如png,jpg
#show_2dimage=True # 是否显示检测结果
show=False # 是否显示检测结果
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

# stereo camera calibration
python stereo_camera_calibration.py \
    --left_file $save_dir/left_cam.yml \
    --right_file $save_dir/right_cam.yml \
    --left_prefix left \
    --right_prefix right \
    --width $width \
    --height $height \
    --left_dir $image_dir_left \
    --right_dir $image_dir_right \
    --image_format  $image_format  \
    --square_size $square_size \
    --save_dir $save_dir \

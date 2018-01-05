# Radar and Lidar fusion using an extended Kalman Filter

This project is to use an extended Kalman Filter to fuse the radar and Lidar sensor measurements for object state estimation. 

# Requirements
* CMake > 2.8

# Uses
* mkdir bin
* mkdir build
* cd build 
* cmake ..
* make
* cd ..
* ./bin/radar_lidar_fusion_EKF ./data/lidar_radar_input.txt ./result/result.txt

# Notes
* Input data format: R/L range azimuth range_rate time_stamp gt_px gt_py gt_vx gt_vy
* Result format: px_estimate py_estimate vx_estimate vy_estimate predicted_px predicted_py gt_px gt_py gt_vx gt_vy


/* 
 * This file defines the measurement package structure 
 * of Lidar and Radar data.
 */
#ifndef RADAR_LIDAR_FUSION_MEASUREMENT_PACKAGE_H_
#define RADAR_LIDAR_FUSION_MEASUREMENT_PACKAGE_H_
#include <stdint.h>
#include <Eigen/Dense>

namespace fusion_ekf
{
enum SensorType
{
  kLIDAR,
  kRADAR
};

struct MeasurementPackage
{
 int64_t timestamp_;
 SensorType sensor_type_;
 Eigen::VectorXd raw_measurements_;
};
}  //namespace fusion_ekf

#endif // RADAR_LIDAR_FUSION_MEASUREMENT_PACKAGE_H_

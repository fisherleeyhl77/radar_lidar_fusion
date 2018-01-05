/* 
 * This file defines the ground truth structure of object states
 * for evaluating the fusion EKF.
 */
#ifndef RADAR_LIDAR_FUSION_GROUND_TRUTH_PACKAGE_H_
#define RADAR_LIDAR_FUSION_GROUND_TRUTH_PACKAGE_H_
#include <stdint.h>
#include <Eigen/Dense>
#include "measurement_package.h"

namespace fusion_ekf
{
struct GroundTruthPackage
{
 int64_t timestamp_;
 SensorType sensor_type_;
 Eigen::VectorXd gt_;
};
}  // namespace fusion_ekf

#endif // RADAR_LIDAR_FUSION_GROUND_TRUTH_PACKAGE_H_

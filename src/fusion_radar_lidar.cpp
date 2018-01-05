#include "fusion_radar_lidar.h"

namespace fusion_ekf
{
FusionEKF::FusionEKF()
{
  is_initialized_ = false;
  previous_timestamp_ = 0;
}

FusionEKF::~FusionEKF() {}

// Initialize the fusion EKF
void FusionEKF::Init(const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0,
	             const MotionModel &motion_model, int64_t timestamp)
{
  ekf_.Init(x0, P0, motion_model);
  previous_timestamp_ = timestamp;
  is_initialized_ = true;
}

// Process the measurement using EKF
void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack,
				   const ObservModel &observ_model)
{
  // Kalman Filter prediction
  float delta_t = 1.0f * (measurement_pack.timestamp_ - 
		  previous_timestamp_) / 1e6;
  ekf_.Predict(delta_t);
  
  // Kalman Filter update
  ekf_.Update(measurement_pack.raw_measurements_, observ_model);
  
  previous_timestamp_ = measurement_pack.timestamp_;
}

// Check whether the EKF is initialized
bool FusionEKF::IsInitialized()
{
  return is_initialized_;
}
}  // namespace fusion_ekf

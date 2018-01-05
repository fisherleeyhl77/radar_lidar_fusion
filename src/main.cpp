#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <Eigen/Dense>

#include "measurement_package.h"
#include "ground_truth_package.h"
#include "fusion_radar_lidar.h"
#include "kalman_filter.h"
#include "tools.h"

// Configurations for the filter
// State transition model
// @param delta_t: sample period
// @param x: state vector (pos_x, pos_y, v_x, v_y)
// Return: predicted state vector and state transition matrix
const auto state_trans_func = [](float delta_t, const Eigen::VectorXd &x) {
  Eigen::MatrixXd F = Eigen::MatrixXd(4, 4);
  // Constant velocity motion model
  F << 1, 0, delta_t, 0,
       0, 1, 0, delta_t,
       0, 0, 1, 0,
       0, 0, 0, 1;
  return std::tuple<Eigen::VectorXd, Eigen::MatrixXd>(F*x, F);
};

// Process noise model
// Standard deviation of continuous process noise (to be tuned to
// get better performance)
const float q_x = 15.0;
const float q_y = 15.0; 
// Compute the covariance matrix of discrete-time process noise
// @param dt: sample period
// @param x: state vector (pos_x, pos_y, v_x, v_y)
// Return: Q matrix
const auto process_noise_func = [](float dt, const Eigen::VectorXd &x) {
  Eigen::MatrixXd Q = Eigen::MatrixXd(4, 4);
  float dt2, dt3, dt4;
  dt2 = dt * dt;
  dt3 = dt2 * dt;
  dt4 = dt3 * dt;

  Q << q_x * dt4 / 4, 0, q_x *dt3 / 2, 0,
       0, q_y * dt4 / 4, 0, q_y * dt3 / 2,
       q_x * dt3 / 2, 0, q_x * dt2, 0,
       0, q_y * dt3 / 2, 0, q_y * dt2;
  return Q;
};

// Motion model that combines state transition model and 
// process noise covariance matrix
const fusion_ekf::MotionModel motion_model = 
    fusion_ekf::MotionModel(state_trans_func, process_noise_func);

// Lidar observation matrix (only measure pos_x and pos_y)
// @param x: state vector
// return: observation matrix
const auto obsev_matrix_func_lidar = [](const Eigen::VectorXd &x) {
  Eigen::MatrixXd H_lidar(2, 4);
  H_lidar << 1, 0, 0, 0,
             0, 1, 0, 0;
  return H_lidar;
};
// Lidar measurement noise cov
// To be determined by lidar noise modeling
const Eigen::MatrixXd R_lidar = (Eigen::MatrixXd(2,2) << 0.05, 0, 
						      0, 0.05).finished();
// Lidar measurement function
// @param x: state vector
// return: lidar measurements (pos_x and pos_y)
const auto lidar_func = [](const Eigen::VectorXd &x) {
  Eigen::VectorXd z(2);
  z << x(0), x(1);
  return z;
};
// Lidar observation model
const fusion_ekf::ObservModel observ_model_lidar(R_lidar, lidar_func, 
				                 obsev_matrix_func_lidar);

// Radar observation matrix
// @param x: state vector
// return: observation matrix
const auto obsev_matrix_func_radar = [](const Eigen::VectorXd &x) {
  Eigen::MatrixXd H_radar(3, 4);
  float px = x(0);
  float py = x(1);
  float vx = x(2);
  float vy = x(3);

  float rho = sqrt(px * px + py * py);
  float rho2 = rho * rho;
  float rho3 = rho2 * rho;

  //check division by zero
  if (rho < 1e-6) {
    std::cerr << "Divide by zero!" << std::endl;
    return H_radar;
  }
  if (abs(rho) < 1e-4) {
    H_radar << 0, 0, 0, 0,
	       0, 0, 0, 0,
	       0, 0, 0, 0;
  } else {
    //compute the Jacobian matrix
    H_radar << px / rho, py / rho, 0, 0,
	       -py / rho2, px / rho2, 0, 0,
	       py * (vx * py - vy * px) / rho3,
	       px * (vy * px - vx * py) / rho3, px / rho, py / rho;
  }
  return H_radar;
};
// Radar measurement noise cov
// To be determined by lidar noise modeling
const Eigen::MatrixXd R_radar = 
    (Eigen::MatrixXd(3, 3) << 0.09, 0, 0,
			      0, 0.16, 0,
                              0, 0, 0.25).finished();
// Radar measurement function
// @param x: state vector
// return: radar measurements (range, azimuth, and range rate)
const auto radar_func = [](const Eigen::VectorXd &x) {
  Eigen::VectorXd z(3);
  float px = x(0);
  float py = x(1);
  float vx = x(2);
  float vy = x(3);
  z << sqrt(px*px + py*py),
       atan2(py, px),
       (px*vx + py*vy)/sqrt(px*px + py*py);
  return z;
};
// Radar observation model
const fusion_ekf::ObservModel observ_model_radar(R_radar, radar_func, 
				                 obsev_matrix_func_radar);

// Initial state error covariance
const Eigen::MatrixXd P_0 = (Eigen::MatrixXd(4,4) << 50, 0, 0, 0,
						     0, 50, 0, 0,
						     0, 0, 100, 0,
				                     0, 0, 0, 100).finished();

// Function prototypes
/**
 * CheckInputs function checks whether the input 
 * command line arguments are valid.
 * If they are invalid, display help info.
 * @param argc Number of comannd line arguments
 * @param argv Array of command line strings
 */
void CheckInputs(int argc, char **argv);

/**
 * CheckFiles function checks whether the input and output files are valid.
 * @param input_file: input file stream
 * @param input_file_name: the input file name
 * @param output_file: output file stream
 * @param output_file_name: the output file name
 */
void CheckFiles(const std::ifstream &input_file, char *input_file_name, 
		const std::ofstream &output_file, char *output_file_name);

/*
 * InitializeEKF function initializes the EKF
 * @param fusion_EKF: Object contains the state estimates after fusion
 * @param measurement_package: storage of lidar and radar measurement
 */
bool InitializeEKF(fusion_ekf::FusionEKF &fusion_EKF, 
                   fusion_ekf::MeasurementPackage &measurement_package);

/*
 * LoadData function loads the measurement and ground truth
 * data from files and stores the data to measurement and ground truth
 * packages
 * @param input_file: input file stream
 * @param measurement_package_vec: vector of lidar and radar measurement
 * packages
 * @param gt_package_vec: vector of ground truth packages
 */
void LoadData(std::ifstream &input_file,
              std::vector<fusion_ekf::MeasurementPackage> 
              &measurement_package_vec,
              std::vector<fusion_ekf::GroundTruthPackage>
              &gt_package_vec);

/*
 * ExecuteFusionEKF function executes the fusion EKF
 * @param measurement_package_vec: vector of lidar and radar measurement
 * packages
 * @param gt_package_vec: vector of ground truth packages
 * @param estimates: vector of state estimates
 * @param ground_truth: vector of ground truth
 * @param output_file: output file stream
 */
void ExecuteFusionEKF(std::vector<fusion_ekf::MeasurementPackage> 
                      &measurement_package_vec,
                      std::vector<fusion_ekf::GroundTruthPackage>
                      &gt_package_vec,
                      std::vector<Eigen::VectorXd> &estimates, 
                      std::vector<Eigen::VectorXd> &ground_truth,
                      std::ofstream &output_file);

/* WriteToFile function writes state estimates, measurements, and ground 
 * truth data to a result file.
 * @param output_file Stream for write data to a file
 * @param fusion_EKF Object contains the state estimates after fusion
 * @param measurement_package_vec Measurement package
 * @param gt_package_vec Ground truth data package
 */
void WriteToFile(std::ofstream &output_file, 
                 const fusion_ekf::FusionEKF &fusion_EKF, 
		 const std::vector<fusion_ekf::MeasurementPackage> 
                 &measurement_package_vec,
		 const std::vector<fusion_ekf::GroundTruthPackage> 
                 &gt_package_vec, 
                 size_t i);

int main(int argc, char **argv)
{
  // Load radar and lidar input data.
  CheckInputs(argc, argv);
  std::ifstream input_file(argv[1], std::ifstream::in);
  std::ofstream output_file(argv[2], std::ofstream::out);
  CheckFiles(input_file, argv[1], output_file, argv[2]);

  // Go through the input data file and store data as vectors of
  // measurement packages and ground truth packages
  std::vector<fusion_ekf::MeasurementPackage> measurement_package_vec;
  std::vector<fusion_ekf::GroundTruthPackage> gt_package_vec;
  LoadData(input_file, measurement_package_vec, gt_package_vec);
  
  // Run EKF for fusing radar and lidar data
  std::vector<Eigen::VectorXd> estimates;
  std::vector<Eigen::VectorXd> ground_truth;
  ExecuteFusionEKF(measurement_package_vec,
                   gt_package_vec, estimates, 
                   ground_truth, output_file);

  // Compute RMSE for accuracy
  std::cout << "\nAccuracy - RMSE:" << std::endl 
       << fusion_ekf::Tools::CalculateRMSE(estimates, ground_truth) 
       << std::endl;

  // close files
  if (output_file.is_open()) {
    output_file.close();
  }
  if (input_file.is_open()) {
    input_file.close();
  } 

  std::cout << "End of main" << std::endl;
  return 0;
}

/**
 * Check whether the command line arguments are valid
 */
void CheckInputs(int argc, char **argv)
{
  std::string help = "Usage: ";
  help += argv[0];
  help += " path/to/input_data.txt path/to/result.txt";

  if (argc == 1) {
    std::cerr << help << std::endl;
  }
  else if (argc == 2) {
    std::cerr << "Please provide a result file.\n" << help << std::endl;
  }
  else if (argc == 3) {
    return;
  }
  else if (argc > 3) {
    std::cerr << "Too many arguments.\n" << help << std::endl;
  }
  exit(EXIT_FAILURE);
}

/**
 * Check whether the input and output files are valid
 */
void CheckFiles(const std::ifstream &input_file, char *input_file_name, 
		const std::ofstream &output_file, char *output_file_name)
{
  if (!input_file.is_open()) {
    std::cerr << "Failed to open the input file: " << 
    input_file_name << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!output_file.is_open()) {
    std::cerr << "Failed to open the output file: " << 
    output_file_name << std::endl;
    exit(EXIT_FAILURE);
  }
}

/*
 * Initialize the EKF for fusion
 * Use the first valid measurement (either radar or lidar) to initialize
 * the estimator.
 */
bool InitializeEKF(fusion_ekf::FusionEKF &fusion_EKF, 
                   fusion_ekf::MeasurementPackage &measurement_pack)
{
  Eigen::VectorXd x_0 = Eigen::VectorXd(4);
  float rho;
  if (measurement_pack.sensor_type_ == fusion_ekf::SensorType::kRADAR) {
    float phi;
    rho = measurement_pack.raw_measurements_[0];
    phi = measurement_pack.raw_measurements_[1];
    x_0 << rho * cos(phi), rho*sin(phi), 0, 0;
  }
  else if (measurement_pack.sensor_type_ == fusion_ekf::SensorType::kLIDAR) {
    float px, py;
    px = measurement_pack.raw_measurements_[0];
    py = measurement_pack.raw_measurements_[1];
    rho = sqrt(px * px + py * py);
    x_0 << px, py, 0, 0;
  }
  // Skip too close measurement
  if (rho < 1e-4) {
    return false;
  }
  fusion_EKF.Init(x_0, P_0, motion_model, measurement_pack.timestamp_);
  return true;
}

/*
 * Load data from files into vectors of measurement package 
 * and ground truth package
 */
void LoadData(std::ifstream &input_file,
              std::vector<fusion_ekf::MeasurementPackage> 
              &measurement_package_vec,
              std::vector<fusion_ekf::GroundTruthPackage>
              &gt_package_vec)
{
  std::string line;
  while (std::getline(input_file, line)) {
    std::istringstream iss(line);
    std::string header, sensor_type;
    iss >> header;
    if (header == "#") {
      std::cout << "Skip the line" << std::endl;
      continue;
    }
    
    // Extract sensor type
    sensor_type = header;    

    // Extract measurements
    fusion_ekf::MeasurementPackage measure_package;
    int64_t timestamp;
    if (sensor_type.compare("L") == 0) {
      // Lidar measurements
      float x;
      float y;
      iss >> x;             // coordinate x 
      iss >> y;             // coordinate y
      iss >> timestamp;     // time stamp
      measure_package.sensor_type_ = fusion_ekf::SensorType::kLIDAR;
      measure_package.timestamp_ = timestamp;
      measure_package.raw_measurements_ = Eigen::VectorXd(2);
      measure_package.raw_measurements_ << x, y;
      measurement_package_vec.push_back(measure_package);
    }
    else if (sensor_type.compare("R") == 0) {
      // Radar measurements
      float rho;
      float phi;
      float rho_dot;
      iss >> rho;           // range
      iss >> phi;           // azimuth
      iss >> rho_dot;       // range rate
      iss >> timestamp;     // time stamp
      measure_package.sensor_type_ = fusion_ekf::SensorType::kRADAR;
      measure_package.timestamp_ = timestamp;
      measure_package.raw_measurements_ = Eigen::VectorXd(3);
      measure_package.raw_measurements_ << rho, phi, rho_dot;
      measurement_package_vec.push_back(measure_package);
    }

    // Load ground truth data and store data to ground truth package
    fusion_ekf::GroundTruthPackage gt_package;
    float gt_px;
    float gt_py;
    float gt_vx;
    float gt_vy;
    iss >> gt_px;           // ground truth x coordinate
    iss >> gt_py;           // ground truth y coordinate
    iss >> gt_vx;           // ground truth x velocity
    iss >> gt_vy;           // ground truth y velocity
    gt_package.gt_ = Eigen::VectorXd(4);
    gt_package.gt_ << gt_px, gt_py, gt_vx, gt_vy;
    gt_package_vec.push_back(gt_package);
  }
}

/*
 * Execute the fusion EKF
 */
void ExecuteFusionEKF(std::vector<fusion_ekf::MeasurementPackage> 
                      &measurement_package_vec,
                      std::vector<fusion_ekf::GroundTruthPackage>
                      &gt_package_vec,
                      std::vector<Eigen::VectorXd> &estimates, 
                      std::vector<Eigen::VectorXd> &ground_truth,
                      std::ofstream &output_file)
{
  fusion_ekf::FusionEKF fusion_EKF;
  size_t N = measurement_package_vec.size();
  for (size_t i = 0; i < N; ++i) {
    if (!fusion_EKF.IsInitialized()) {
      bool result = InitializeEKF(fusion_EKF, measurement_package_vec[i]);
      if (!result) {
	std::cout << "Skip frame" << i << std::endl;
	continue;
      }
    }
    // Filter data using EKF
    if (measurement_package_vec[i].sensor_type_ == 
        fusion_ekf::SensorType::kRADAR) {
      fusion_EKF.ProcessMeasurement(measurement_package_vec[i], 
				    observ_model_radar);
    }
    else if (measurement_package_vec[i].sensor_type_ == 
        fusion_ekf::SensorType::kLIDAR) {
      fusion_EKF.ProcessMeasurement(measurement_package_vec[i], 
				    observ_model_lidar);
    }

    // Store the results to file
    WriteToFile(output_file, fusion_EKF, measurement_package_vec, 
		gt_package_vec, i);

    estimates.push_back(fusion_EKF.ekf_.x_);
    ground_truth.push_back(gt_package_vec[i].gt_);
  }
}

/*
 * Write result to the output file
 */
void WriteToFile(std::ofstream &output_file, 
                 const fusion_ekf::FusionEKF &fusion_EKF, 
		 const std::vector<fusion_ekf::MeasurementPackage> 
                 &measurement_package_vec, 
		 const std::vector<fusion_ekf::GroundTruthPackage> 
                 &gt_package_vec, size_t i)
{
  // Output the state estimates
  output_file << fusion_EKF.ekf_.x_(0) << "\t";
  output_file << fusion_EKF.ekf_.x_(1) << "\t";
  output_file << fusion_EKF.ekf_.x_(2) << "\t";
  output_file << fusion_EKF.ekf_.x_(3) << "\t";

  // Output the predicted states from measurements
  if (measurement_package_vec[i].sensor_type_ == 
      fusion_ekf::SensorType::kLIDAR) {
    // Output the estimation
    output_file << measurement_package_vec[i].raw_measurements_(0) << "\t";
    output_file << measurement_package_vec[i].raw_measurements_(1) << "\t";
  } else if (measurement_package_vec[i].sensor_type_ == 
      fusion_ekf::SensorType::kRADAR) {
    // Output the estimation in the cartesian coordinates
    float rho = measurement_package_vec[i].raw_measurements_(0);
    float phi = measurement_package_vec[i].raw_measurements_(1);
    output_file << rho * cos(phi) << "\t";
    output_file << rho * sin(phi) << "\t";
  }

  // Output the ground truth packages
  output_file << gt_package_vec[i].gt_(0) << "\t";
  output_file << gt_package_vec[i].gt_(1) << "\t";
  output_file << gt_package_vec[i].gt_(2) << "\t";
  output_file << gt_package_vec[i].gt_(3) << "\n";
}

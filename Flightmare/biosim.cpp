#include <ros/ros.h>
#include <std_msgs/String.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/bridges/unity_message_types.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/objects/quadrotor.hpp"

using namespace flightlib;

std::string current_movement = "";
void movementCallback(const std_msgs::String::ConstPtr& msg) {
  current_movement = msg->data;
}

void updateMovement(const ros::TimerEvent& event, std::shared_ptr<Quadrotor> quad_ptr) {
  QuadState quad_state;
  quad_ptr->getState(&quad_state);

  float move = 0.05f;

  // Get the rotation matrix from the current orientation
  Eigen::Quaterniond q(quad_state.x[QS::ATTW], quad_state.x[QS::ATTX], quad_state.x[QS::ATTY], quad_state.x[QS::ATTZ]);
  Eigen::Matrix3d R = q.toRotationMatrix();

  // move vector (non rotation)
  Eigen::Vector3d movement(0, 0, 0);
  if (current_movement == "left") {
    movement.x() = -move;
  } else if (current_movement == "down") {
    movement.z() = -move;
  } else if (current_movement == "right") {
    movement.x() = move;
  } else if (current_movement == "backward") {
    movement.y() = -move;
  } else if (current_movement == "forward") {
    movement.y() = move;
  } else if (current_movement == "up") {
    movement.z() = move;
  }

  // Rotate movement vector to map orient
  Eigen::Vector3d world_movement = R * movement;

  // quadstate updating
  quad_state.x[QS::POSX] += world_movement.x();
  quad_state.x[QS::POSY] += world_movement.y();
  quad_state.x[QS::POSZ] += world_movement.z();

  // rotations  (10 is just arbitrary speed)
  if (current_movement == "rotate_up") {
    Eigen::AngleAxisd rotate_angle(10 * move * M_PI / 180.0, Eigen::Vector3d::UnitX());
    Eigen::Quaterniond q_rot(rotate_angle);
    Eigen::Quaterniond q_new = q * q_rot;
    quad_state.x[QS::ATTW] = q_new.w();
    quad_state.x[QS::ATTX] = q_new.x();
    quad_state.x[QS::ATTY] = q_new.y();
    quad_state.x[QS::ATTZ] = q_new.z();
  } else if (current_movement == "rotate_down") {
    Eigen::AngleAxisd rotate_angle(-10 * move * M_PI / 180.0, Eigen::Vector3d::UnitX());
    Eigen::Quaterniond q_rot(rotate_angle);
    Eigen::Quaterniond q_new = q * q_rot;
    quad_state.x[QS::ATTW] = q_new.w();
    quad_state.x[QS::ATTX] = q_new.x();
    quad_state.x[QS::ATTY] = q_new.y();
    quad_state.x[QS::ATTZ] = q_new.z();
  } else if (current_movement == "rotate_left") {
    Eigen::AngleAxisd rotate_angle(10 * move * M_PI / 180.0, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_rot(rotate_angle);
    Eigen::Quaterniond q_new = q * q_rot;
    quad_state.x[QS::ATTW] = q_new.w();
    quad_state.x[QS::ATTX] = q_new.x();
    quad_state.x[QS::ATTY] = q_new.y();
    quad_state.x[QS::ATTZ] = q_new.z();
  } else if (current_movement == "rotate_right") {
    Eigen::AngleAxisd rotate_angle(-10 * move * M_PI / 180.0, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_rot(rotate_angle);
    Eigen::Quaterniond q_new = q * q_rot;
    quad_state.x[QS::ATTW] = q_new.w();
    quad_state.x[QS::ATTX] = q_new.x();
    quad_state.x[QS::ATTY] = q_new.y();
    quad_state.x[QS::ATTZ] = q_new.z();
  } else if (current_movement == "twistleft") {
    Eigen::AngleAxisd rotate_angle(-10 * move * M_PI / 180.0, Eigen::Vector3d::UnitY());
    Eigen::Quaterniond q_rot(rotate_angle);
    Eigen::Quaterniond q_new = q * q_rot;
    quad_state.x[QS::ATTW] = q_new.w();
    quad_state.x[QS::ATTX] = q_new.x();
    quad_state.x[QS::ATTY] = q_new.y();
    quad_state.x[QS::ATTZ] = q_new.z();
  } else if (current_movement == "twistright") {
    Eigen::AngleAxisd rotate_angle(10 * move * M_PI / 180.0, Eigen::Vector3d::UnitY());
    Eigen::Quaterniond q_rot(rotate_angle);
    Eigen::Quaterniond q_new = q * q_rot;
    quad_state.x[QS::ATTW] = q_new.w();
    quad_state.x[QS::ATTX] = q_new.x();
    quad_state.x[QS::ATTY] = q_new.y();
    quad_state.x[QS::ATTZ] = q_new.z();
  }

  quad_ptr->setState(quad_state);
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "quad_movement_example");
  ros::NodeHandle nh("");
  ros::NodeHandle pnh("~");

  // unity quadrotor
  std::shared_ptr<Quadrotor> quad_ptr = std::make_shared<Quadrotor>();
  Vector<3> quad_size(0.5, 0.5, 0.5);
  quad_ptr->setSize(quad_size);
  QuadState quad_state;

  // Flightmare(Unity3D)
  std::shared_ptr<UnityBridge> unity_bridge_ptr = UnityBridge::getInstance();
  SceneID scene_id{UnityScene::WAREHOUSE};
  bool unity_ready{false};

  // ROS subscriber
  ros::Subscriber sub = nh.subscribe<std_msgs::String>("COMMAND", 10, movementCallback);

  // ROS movement updater
  ros::Timer timer = nh.createTimer(ros::Duration(0.01), boost::bind(updateMovement, _1, quad_ptr));

  // initialization
  quad_state.setZero();
  quad_ptr->reset(quad_state);

  // connect unity
  unity_bridge_ptr->addQuadrotor(quad_ptr);
  unity_ready = unity_bridge_ptr->connectUnity(scene_id);

  FrameID frame_id = 0;
  while (ros::ok() && unity_ready) {
    ros::spinOnce();

    unity_bridge_ptr->getRender(frame_id);
    unity_bridge_ptr->handleOutput();

    frame_id += 1;
  }

  return 0;
}


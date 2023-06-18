#include <ros/ros.h>
#include <std_msgs/String.h>
#include <vector>
#include <random>

int main(int argc, char *argv[]) {

  ros::init(argc, argv, "movement_publisher");
  ros::NodeHandle nh;

  ros::Publisher movement_pub = nh.advertise<std_msgs::String>("movement", 10);

  std::vector<std::string> movements = {
      "backward", "up", "up", "down", "fist", "forward", "left", "resting",
      "right", "rotate_down", "up" ,"up", "rotate_left", "rotate_right", "rotate_up",
      "stop", "twistleft", "twistright", "up"};


  ros::Rate loop_rate(1.0 / 0.5);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, movements.size() - 1);

  while (ros::ok()) {
    std_msgs::String msg;
    int movement_idx = dis(gen);
    msg.data = movements[movement_idx];
    movement_pub.publish(msg);
    loop_rate.sleep();
  }

  return 0;
}


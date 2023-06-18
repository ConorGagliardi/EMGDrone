System requirements: 
Linux distribution such as Ubuntu 20.04 (WSL recommended: https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-10#1-overview)
ROS (Noetic recommended: http://wiki.ros.org/noetic/Installation/Ubuntu)
ros_myo package (https://github.com/uts-magic-lab/ros_myo)
Flightmare (https://flightmare.readthedocs.io/en/latest/getting_started/quick_start.html)

Assuming you have your Flightmare workspace set up:
1. Create a 'biosim' folder inside of ws/src/flightmare/flightros/src, and place the biosim.cpp and
mtest.cpp files in that folder.
2. Create another 'biosim' folder inside of ws/src/flightmare/flightros/launch, and place the
biosim.launch and mtest.launch files in that folder.
3. Replace the CMakeLists.txt file in ws/src/flightmare/flightros with the version provided.
4. Run 'catkin build' from ws

To run the simulation:
Just run biosim.launch using 'roslaunch flightros biosim.launch'. The simualtion will open, and
if the ros_myo scripts are already running, the quadcopter will begin moving according to the
live gesture classifications made by the ML model.

biosim.cpp listens for gesture commands coming from a /COMMAND rostopic by default, which is the
rostopic that myoOutput.py publishes to.
If you would like to see the simulation run standalone using a preconfigured gesture pattern:
1. Edit biosim.cpp. On the line under //ROS subscriber, change the string from "COMMAND" to "movement"
2. Rebuild the workspace from the ws folder using 'catkin build'
3. Relaunch the simulation, then run mtest using 'roslaunch flightros mtest.launch'
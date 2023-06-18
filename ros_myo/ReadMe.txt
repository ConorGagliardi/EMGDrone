System requirements: 
Linux distribution such as Ubuntu 20.04 (WSL recommended: https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-10#1-overview)
ROS (Noetic recommended: http://wiki.ros.org/noetic/Installation/Ubuntu)
ros_myo package (https://github.com/uts-magic-lab/ros_myo)
Flightmare (https://flightmare.readthedocs.io/en/latest/getting_started/quick_start.html)


The programs myo-rawNode.py and myoOutput.py are designed to be run within a ROS environment. 
You must have the package ros_myo installed. You can place these scripts in the 'scripts' file that 
comes with the package when you clone it from here: https://github.com/uts-magic-lab/ros_myo

Create a workspace for this package using 'catkin make' in the Linux terminal. 
Replace the myo-rawNode.py file that comes with it with the modified version contained in this directory. 
Additionally, place the raw data directories in the scripts directory inside the ros_myo workspace.
Edit the datapath line in myoOutput.py to reflect your workspace's name and folder structure.
Add the Aharon_Fusion_Model folder to the scripts directory.

To run the real-time classifier: 
1. If not already done, run source devel/setup.bash at the highest level in your workspace. 
2. Run myo-rawNode.py first - this establishes a connection to a Myo armband if you have a Bluetooth 
dongle inserted. If you connected your Myo to your computer through Windows and your environment with
ros_myo and Flightmare is in WSL, you will need to connect your Myo bluetooth dongle to WSL using the
instructions found here: https://learn.microsoft.com/en-us/windows/wsl/connect-usb.
You may also need to run sudo chmod a+rw /dev/ttyACM0 first to get access to the USB port.
3. Run myoOutput.py. This program runs SVM training first and then begins publishing classifications
based on the gesture made at the prompt "Start!" in the command line. You should see the result 
after the accumulation window closes.



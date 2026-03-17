# Source the setup.bash file for the second ROS workspace
source /root/online_rl/catkin_ws/devel/setup.bash

# Change the ROS master URI to a different port
export ROS_MASTER_URI=http://localhost:11511

# Run the second instance of franka_server.py in the background
python franka_server.py \
    --robot_ip=192.168.1.221 \
    --gripper_type=Franka \
    --flask_url=127.0.0.2 \
    --ros_port=11511
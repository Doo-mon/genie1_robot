
conda activate myenv5090

# bash /home/zhailinbo/ZJAY/RL/robot/real_robot_g1/workspace_zzh/env_for_g1.sh

# ping 10.42.0.101
cd /home/zhailinbo/ZJAY/RL/robot
cd a2d_sdk
source env.sh
cd ..

unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset PYTHONPATH


cd real_robot_g1/workspace_zzh
echo "cd real_robot_g1/workspace_zzh"

echo "[INFO] Finished, you can control robot now."
echo "[EXAMPLE]  bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_rollout_rtc.sh "



# cd /home/user/robot/hrl-kdl-indigo-devel/pykdl_utils
# pip install . --no-build-isolation


# pip install rospy --extra-index-url https://rospypi.github.io/simple/

# -i https://pypi.tuna.tsinghua.edu.cn/simple

# python -m pip install -U pip setuptools wheel
# python -m pip install \
#   rospy rosbag roslib roslz4 \
#   --extra-index-url https://rospypi.github.io/simple/ 


# sensor_msgs
# pytest

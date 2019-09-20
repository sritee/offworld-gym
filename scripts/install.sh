#!/usr/bin/env bash

# this assumes the script is executed from under `scripts` directory
# (should we move `install.sh` to the repository root? will be cleaner)
cd ..
export OFFWORLD_GYM_ROOT=`pwd`

# make sure we have Python 3.5
sudo apt update
sudo apt install -y python3.5 python3.5-dev libbullet-dev

pip install --user numpy
pip install --user tensorflow-gpu
pip install --user keras==2.2.4
pip install --user opencv-python
pip install --user catkin_pkg
pip install --user empy
pip install --user requests
pip install --user defusedxml
pip install --user rospkg
pip install --user matplotlib
pip install --user netifaces
pip install --user regex
pip install --user psutil
pip install --user gym
pip install --user python-socketio
pip install --user scikit-image
cd $OFFWORLD_GYM_ROOT
pip install --user -e .

# Python3.6
sudo su
cd /opt
wget https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tgz
tar -xvf Python-3.6.3.tgz
cd Python-3.6.3
./configure
make
make install
curl https://bootstrap.pypa.io/get-pip.py | sudo -H python3.6
exit

pip3.6 install --user setuptools
pip3.6 install --user numpy
pip3.6 install --user tensorflow-gpu
pip3.6 install --user keras==2.2.4
pip3.6 install --user opencv-python
pip3.6 install --user catkin_pkg
pip3.6 install --user empy
pip3.6 install --user requests
pip3.6 install --user defusedxml
pip3.6 install --user matplotlib
pip3.6 install --user netifaces
pip3.6 install --user regex
pip3.6 install --user psutil
pip3.6 install --user gym
pip3.6 install --user python-socketio
pip3.6 install --user scikit-image
cd $OFFWORLD_GYM_ROOT
pip3.6 install --user -e .

source /opt/ros/kinetic/setup.bash

pip install -e .

# install additional ROS packages
sudo apt install -y ros-kinetic-grid-map ros-kinetic-frontier-exploration \
                    ros-kinetic-ros-controllers ros-kinetic-rospack \
                    libignition-math2 libignition-math2-dev python3-tk libeigen3-dev \
                    ros-kinetic-roslint


# Milestone 1: Python and system packages
if [ $? -eq 0 ]
then
  printf "\nOK: Python and system packages were installed successfully.\n\n"
else
  printf "\nFAIL: Errors detected installing system packages, please resolve them and restart the installation script.\n\n" >&2
  exit 1
fi

cd $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src

git clone https://github.com/ros/geometry2.git -b indigo-devel
git clone https://github.com/ros-simulation/gazebo_ros_pkgs.git -b kinetic-devel
git clone https://github.com/ros-perception/vision_opencv.git -b kinetic
git clone https://github.com/offworld-projects/offworld_rosbot_description.git -b kinetic-devel
cd ..
catkin_make

# Milestone 1: Python and system packages
if [ $? -eq 0 ]
then
  printf "\nOK: ROS workspace built successfully\n\n"
else
  printf "\nFAIL: Errors detected while building ROS workspace. Please resolve the issues and finish installation manually, line-by-line, do no restart this script.\n\n" >&2
  exit 1
fi

echo "ROS dependencies build complete."

# build the Gym Shell script
echo '#!/usr/bin/env bash' > $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "source ~/ve/py35gym/bin/activate" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "unset PYTHONPATH" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "source /opt/ros/kinetic/setup.bash" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "source $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/devel/setup.bash --extend" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "export GAZEBO_MODEL_PATH=$OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/models:$GAZEBO_MODEL_PATH" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo 'export PYTHONPATH=~/ve/py35gym/lib/python3.5/site-packages:$PYTHONPATH' >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "export OFFWORLD_GYM_ROOT=$OFFWORLD_GYM_ROOT" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
chmod +x $OFFWORLD_GYM_ROOT/scripts/gymshell.sh

# update to gazebo 7.13
# http://answers.gazebosim.org/question/18934/kinect-in-gazebo-not-publishing-topics/
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
sudo apt install wget
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y gazebo7 libgazebo7-dev

printf "\n\nInstallation complete\n---------------------\n\n"
printf "To setup a shell for OffWorld Gym run\n\n\tsource $OFFWORLD_GYM_ROOT/scripts/gymshell.sh\n\nin each new terminal to activate Gym Shell.\n"
printf "Or add to your ~/.bashrc by running\n\n\techo \"source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh\" >> ~/.bashrc\n\n---------------------\n\n"
printf "To test Real environment:\n\t(add instructions here)\n\n"
printf "To test Sim environment: open two terminals, activate Gym Shell, and run:\n\t1. roslaunch gym_offworld_monolith env_bringup.launch\n\t2. gzclient\n\n"

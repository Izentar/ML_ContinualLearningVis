#!/bin/bash
set -e

echo "do not invoke this script"
echo "use only needed commands"
echo "use on your own risk !!!"

exit 1


sudo apt update
sudo apt install -y python3 python3-pip python3-venv 
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 2
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3 2
sudo apt install -y ubuntu-drivers-common
sudo apt install -y alsa-utils
sudo ubuntu-drivers install

# if segmentation fault occurs when importing libs
# remove old pythonEnv and create new after creating symlink
# choose appropriate version of apt_pkg.cpython*
sudo ln -s apt_pkg.cpython-310-x86_64-linux-gnu.so apt_pkg.so 
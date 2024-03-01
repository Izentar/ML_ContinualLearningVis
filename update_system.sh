#!/bin/bash
set -e

# use on your own risk !!!

sudo apt update
sudo apt install -y python3.11 python3-pip python3.11-venv 
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 2
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
sudo apt install -y ubuntu-drivers-common
sudo apt install -y alsa-utils
sudo ubuntu-drivers install
#!/bin/bash
set -e

sudo apt update
sudo apt install -y python3.11 python3-pip python3.11-venv 
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
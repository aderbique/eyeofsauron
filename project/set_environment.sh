#!/bin/bash

sudo apt update -y && \
sudo apt install -y software-properties-common && \
sudo apt install -y python3.9 python3.9-venv && \
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

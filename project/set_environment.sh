#!/bin/bash

sudo apt update -y && \
sudo apt install -y software-properties-common && \
sudo apt install -y python3.9 python3.9-venv && \
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo "Obtaining Data Sets"
mkdir -p datasets
echo "Downloading KDD Dataset for Task 1"
wget http://205.174.165.80/CICDataset/NSL-KDD/Dataset/NSL-KDD.zip -P datasets/
echo "Extracting KDD Dataset for Task 1"
unzip datasets/NSL-KDD.zip -d datasets/kdd

echo "Downloading KDD Dataset for Task 3"
wget http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip -P datasets/
echo "Extracting CIC-IDS-2017 Dataset"
unzip datasets/MachineLearningCSV.zip -d datasets/ids
rm -f datasets/{MachineLearningCSV.zip,NSL-KDD.zip}
echo "Setup Complete"

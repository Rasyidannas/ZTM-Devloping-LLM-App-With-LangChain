#!/usr/bin/bash

# Create a new conda environment named 'ztm'
conda create --name ztm python=3.11 -y

# Activate the environment
conda activate ztm

# Install requirements
pip install -r requirements.txt

# Add any additional setup steps below
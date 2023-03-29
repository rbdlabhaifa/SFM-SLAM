# Structure from Motion (SfM) Pipeline

This project implements a Structure from Motion (SfM) pipeline to reconstruct a 3D point cloud and estimate camera poses from a set of overlapping images.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [SfM Pipeline Overview](#sfm-pipeline-overview)
- [Customization](#customization)

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Install Python 3.x from the official [Python website](https://www.python.org/downloads/).
2. Install OpenCV and NumPy using pip:

```bash
pip install opencv-python opencv-contrib-python numpy

Usage
Import the StructureFromMotion class from the sfm.py file.
Create an instance of the class with the input images and the desired feature detector.
Call the run method to execute the SfM pipeline.
Example:

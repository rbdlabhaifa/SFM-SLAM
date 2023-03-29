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
```
Usage
Import the StructureFromMotion class from the sfm.py file.
Create an instance of the class with the input images and the desired feature detector.
Call the run method to execute the SfM pipeline.
Example:
```
from sfm import StructureFromMotion

images = [...] # Load your images here as a list of NumPy arrays
sfm = StructureFromMotion(images, feature_detector='SIFT')
sfm.run()
```

## SfM Pipeline Overview

The SfM pipeline consists of the following steps:
Detect and describe keypoints in each image.
Match keypoints across pairs of images.
Robustly estimate matches using RANSAC or another robust estimation technique.
Initialize the 3D structure and camera poses with a suitable pair of images.
Incrementally add images to the reconstruction by:
Estimating camera pose for each new image.
Triangulating new 3D points from matched keypoints.
Updating existing 3D points with new observations.
Applying bundle adjustment to refine the updated reconstruction.
Optionally, detect loop closures and apply global optimization to improve the overall consistency of the reconstruction.

## Customization

You can customize the SfM pipeline by modifying the methods in the StructureFromMotion class. This can include changing the feature detector, matching algorithm, or robust estimation technique, as well as adding new functionality or optimization steps.


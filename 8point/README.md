# Structure From Motion (SfM) Demonstration
## Introduction
This code provides a demonstration of the Structure from Motion (SfM) technique, a method used to recover the 3D structure of a scene from multiple 2D images. By using multiple views of a scene, SfM algorithms estimate camera poses and a sparse set of 3D points that define the observed scene.

## Features
* StructureFromMotion Class: This class serves as the core of our SfM algorithm. The main methods of this class are:

    * best_R_T: Determines the best rotation (R) and translation (T) matrices by minimizing reprojection errors.
    * use8point_opencv: Uses OpenCV's built-in functions to determine R and T using the 8-point algorithm.
    * use8point: Implements the 8-point algorithm manually to compute the essential matrix and subsequently derive R and T.
* Reprojection Error Calculation: The code computes the reprojection error, which measures the difference between the actual 2D point and its reprojection using estimated R and T. Lower reprojection errors indicate a better fit of estimated camera poses and 3D points.

* Visualization: The code visualizes the randomly generated 3D points and their 2D projections in two images: the original and the transformed one.

## Usage
The script starts by generating a set of random 3D points. It then defines a fixed rotation and translation to simulate the transformation of the camera between two views. Using these fixed values, it projects the 3D points onto two 2D planes.

The script then applies the SfM methods to estimate the R and T from the 2D projections. Both the OpenCV-based method and the manual implementation of the 8-point algorithm are employed.

After obtaining the estimated R and T, the script computes and prints the errors between the true and estimated values. Additionally, it computes the reprojection error for both methods, helping to evaluate the accuracy of the estimation.

Finally, the script visualizes the 3D structure and the two 2D projections in a series of plots.

## Dependencies
* NumPy: For numerical operations and matrix manipulations.
* OpenCV: Used for the 8-point algorithm and other computer vision functionalities.
* Matplotlib: For visualizing the 3D points and their 2D projections.
## Conclusion
This code offers an insight into the fundamental operations behind Structure from Motion, demonstrating how to estimate camera poses and evaluate the accuracy of these estimations. By visualizing the results, one can gain an intuitive understanding of the SfM process.
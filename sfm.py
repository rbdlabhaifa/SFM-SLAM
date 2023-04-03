import cv2
import numpy as np


class StructureFromMotion:
    def __init__(self, images, feature_detector='SIFT'):
        self.images = images
        self.feature_detector = feature_detector

    def detect_and_describe_keypoints(self):
        """
        Detect keypoints in each image and compute their descriptors using the specified feature detector.
        
        Inputs: None (uses self.images and self.feature_detector)
        Outputs: keypoints_list, descriptors_list (lists of keypoints and descriptors for each image)
        """
        pass

    def match_keypoints(self, descriptors_list):
        """
        Match keypoints across pairs of images using their descriptors.
        
        Inputs: descriptors_list (list of descriptors for each image)
        Outputs: matches_list (list of matches for each pair of images)
        """
        pass

    def robustly_estimate_matches(self, matches_list):
        """
        Apply a robust estimation technique (e.g., RANSAC) to remove outlier matches.
        
        Inputs: matches_list (list of matches for each pair of images)
        Outputs: filtered_matches_list (list of filtered matches for each pair of images)
        """
        pass

    def best_R_T(self, points1, points2, R, T, K):
        # Choose R, T with lowest reprojection error
        min_error = float('inf')
        best_R = None
        best_T = None

        for i in range(len(R)):
            P1 = K @ np.eye(3, 4)
            P2 = K @ np.hstack((R[i], T[i][:, np.newaxis]))

            # Compute reprojection error
            error = 0
            for j in range(len(points1)):
                x1_hom = np.hstack((points1[j], [1]))
                x2_hom = np.hstack((points2[j], [1]))
                x1_reproj = P1 @ x1_hom
                x2_reproj = P2 @ x2_hom
                x1_reproj /= x1_reproj[-1]
                x2_reproj /= x2_reproj[-1]
                error += np.sum((x1_hom[:2] - x1_reproj[:2]) ** 2)
                error += np.sum((x2_hom[:2] - x2_reproj[:2]) ** 2)
            error /= (2 * len(points1))

            if error < min_error:
                min_error = error
                best_R = R[i]
                best_T = T[i]

        return best_R, best_T

    def use8point_opencv(self, points1, points2, K=None):
        """
        Initialize camera poses with a suitable pair of images.
        Input: points1 (set of points from first image)
               points2 (set of matching points from second image)
               K (Intrinsic camera matrix)
        Outputs: Rotation and Translation from first camera to second camera
        """

        # Define intrinsic camera matrix (assuming unit focal length)
        if K is None:
            K = np.eye(3)

        # Compute the essential matrix from the camera matrix and point correspondences
        E, _ = cv2.findEssentialMat(points1, points2, K, method=cv2.FM_8POINT, threshold=0.1)

        _, R, t, _ = cv2.recoverPose(E, points1, points2, K)

        return R, t

    def use8point(self, points1, points2, K=None):
        """
        Initialize camera poses with a suitable pair of images.
        Input: points1 (set of points from first image)
               points2 (set of matching points from second image)
               K (Intrinsic camera matrix)
        Outputs: Rotation and Translation from first camera to second camera
        """

        # Define intrinsic camera matrix (assuming unit focal length)
        if K is None:
            K = np.eye(3)

        x1 = points1[:, 0]
        y1 = points1[:, 1]
        x1dash = points2[:, 0]
        y1dash = points2[:, 1]
        A = np.zeros((len(x1), 9))
        for i in range(len(x1)):
            A[i] = np.array(
                [x1dash[i] * x1[i], x1dash[i] * y1[i], x1dash[i], y1dash[i] * x1[i], y1dash[i] * y1[i], y1dash[i],
                 x1[i], y1[i], 1])

        # taking SVD of A for estimation of F
        U, S, V = np.linalg.svd(A, full_matrices=True)
        F_est = V[-1, :]
        F_est = F_est.reshape(3, 3)

        # Enforcing rank 2 for F
        ua, sa, va = np.linalg.svd(F_est, full_matrices=True)
        sa = np.diag(sa)
        sa[2, 2] = 0
        F = np.dot(ua, np.dot(sa, va))
        E_est = np.dot(K.T, np.dot(F, K))

        # reconstructing E by correcting singular values
        U, S, V = np.linalg.svd(E_est, full_matrices=True)
        S = np.diag(S)
        S[0, 0], S[1, 1], S[2, 2] = 1, 1, 0
        E = np.dot(U, np.dot(S, V))

        U, S, V = np.linalg.svd(E)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        # Generate candidate R, T
        R1 = np.dot(U, np.dot(W, V))
        R2 = np.dot(U, np.dot(W, V))
        R3 = np.dot(U, np.dot(W.T, V))
        R4 = np.dot(U, np.dot(W.T, V))

        T1 = U[:, 2]
        T2 = -U[:, 2]
        T3 = U[:, 2]
        T4 = -U[:, 2]

        R = [R1, R2, R3, R4]
        T = [T1, T2, T3, T4]

        for i in range(len(R)):
            if (np.linalg.det(R[i]) < 0):
                R[i] = -R[i]
                T[i] = -T[i]

        R, T = self.best_R_T(R, T, K)

        return R, T

    def initialize_structure(self, keypoints_list, filtered_matches_list):
        """
        Initialize the 3D structure and camera poses with a suitable pair of images.
        
        Inputs: keypoints_list (list of keypoints for each image),
                filtered_matches_list (list of filtered matches for each pair of images)
        Outputs: initial_structure (initial 3D point cloud),
                 initial_camera_poses (initial camera poses for the chosen pair of images)
        """
        pass

    def estimate_camera_pose(self, keypoints, filtered_matches, structure):
        """
        Estimate the camera pose for a new image using PnP (Perspective-n-Point) algorithms.
        
        Inputs: keypoints (keypoints of the new image),
                filtered_matches (filtered matches between the new image and the existing structure),
                structure (existing 3D point cloud)
        Outputs: camera_pose (estimated camera pose for the new image)
        """
        pass

    def triangulate_new_points(self, keypoints1, keypoints2, camera_pose1, camera_pose2, matches):
        """
        Triangulate new 3D points from matched keypoints in the new image and the existing 3D structure.
        
        Inputs: keypoints1 (keypoints of the new image),
                keypoints2 (keypoints of the existing structure),
                camera_pose1 (camera pose of the new image),
                camera_pose2 (camera pose of the existing structure),
                matches (filtered matches between the new image and the existing structure)
        Outputs: new_points (list of newly triangulated 3D points)
        """
        pass

    def update_existing_points(self, keypoints1, keypoints2, matches, structure):
        """
        Update the existing 3D points with new observations from the new image.
        
        Inputs: keypoints1 (keypoints of the new image),
                keypoints2 (keypoints of the existing structure),
                matches (filtered matches between the new image and the existing structure),
                structure (existing 3D point cloud)
        Outputs: updated_structure (updated 3D point cloud)
        """
        pass

    def bundle_adjustment(self, camera_poses, structure, keypoints_list, filtered_matches_list):
        """
        Apply bundle adjustment to refine the 3D structure and camera poses by minimizing the reprojection error.
        
        Inputs: camera_poses (list of camera poses),
                structure (current 3D point cloud),
                keypoints_list (list of keypoints for each image),
                filtered_matches_list (list of filtered matches for each pair of images)
        Outputs: refined_camera_poses (refined list of camera poses),
                 refined_structure (refined 3D point cloud)
        """
        pass

    def detect_loop_closure(self, keypoints_list, descriptors_list, camera_poses):
        """
        Detect loop closures, where the camera revisits a previously observed part of the scene.

        Inputs: keypoints_list (list of keypoints for each image),
                descriptors_list (list of descriptors for each image),
                camera_poses (list of camera poses)
        Outputs: loop_closures (list of detected loop closures, each represented by a tuple of image indices)
        """
        pass

    def global_optimization(self, camera_poses, structure, loop_closures):
        """
        Apply global optimization techniques (e.g., global bundle adjustment or pose graph optimization) to correct
        for drift and improve the overall consistency of the reconstruction.

        Inputs: camera_poses (list of camera poses),
                structure (current 3D point cloud),
                loop_closures (list of detected loop closures)
        Outputs: optimized_camera_poses (optimized list of camera poses),
                 optimized_structure (optimized 3D point cloud)
        """
        pass

    def run(self):
        """
        Execute the SfM pipeline by incrementally reconstructing the 3D structure and updating the camera poses.

        Inputs: None (uses self.images and self.feature_detector)
        Outputs: final_camera_poses (final list of camera poses),
                 final_structure (final 3D point cloud)
        """
        # Step 1: Detect and describe keypoints
        keypoints_list, descriptors_list = self.detect_and_describe_keypoints()

        # Step 2: Match keypoints across pairs of images
        matches_list = self.match_keypoints(descriptors_list)

        # Step 3: Robustly estimate matches
        filtered_matches_list = self.robustly_estimate_matches(matches_list)

        # Step 4: Initialize the 3D structure and camera poses
        initial_structure, initial_camera_poses = self.initialize_structure(keypoints_list, filtered_matches_list)

        # Initialize the final structure and camera poses with the initial values
        final_structure = initial_structure
        final_camera_poses = initial_camera_poses

        # Iterate through the remaining images
        for i in range(len(self.images)):
            if i not in initial_camera_poses:
                # Step 5a: Estimate the camera pose for the new image
                camera_pose = self.estimate_camera_pose(keypoints_list[i], filtered_matches_list[i], final_structure)

                # Step 5b: Triangulate new points
                new_points = self.triangulate_new_points(
                    keypoints_list[i], keypoints_list[final_camera_poses.keys()[0]],
                    camera_pose, final_camera_poses[final_camera_poses.keys()[0]], filtered_matches_list[i]
                )

                # Step 5c: Update existing points
                updated_structure = self.update_existing_points(
                    keypoints_list[i], keypoints_list[final_camera_poses.keys()[0]],
                    filtered_matches_list[i], final_structure
                )

                # Step 5d: Bundle adjustment
                refined_camera_poses, refined_structure = self.bundle_adjustment(
                    final_camera_poses, updated_structure, keypoints_list, filtered_matches_list
                )

                # Update the final structure and camera poses with the refined values
                final_structure = refined_structure
                final_camera_poses = refined_camera_poses

        # Step 6: Detect loop closures and apply global optimization (optional)
        loop_closures = self.detect_loop_closure(keypoints_list, descriptors_list, final_camera_poses)
        optimized_camera_poses, optimized_structure = self.global_optimization(
            final_camera_poses, final_structure, loop_closures
        )

        # Update the final structure and camera poses with the optimized values
        final_structure = optimized_structure
        final_camera_poses = optimized_camera_poses

        return final_camera_poses, final_structure

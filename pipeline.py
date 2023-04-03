import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

class StructureFromMotion:
    def __init__(self, images, feature_detector='SIFT'):
        self.images = images
        self.feature_detector = feature_detector

# ======================================================================================================

    def find_perfect_match(self, set1, set2):
        """
           Finds the optimal matching between two sets of 3D points and visualizes the results using a table plot and a 3D scatter plot.
           Optimal matching = minimal sum of Euclidean distances between two sets of points.

           Args:
           - set1: A numpy array of shape (n_samples, n_features) representing the first set of points.
           - set2: A numpy array of shape (n_samples, n_features) representing the second set of points with R,T implementation.


           Returns:
           - The function returns two dictionaries numerated_set1 and numerated_set2, which represent the optimal matching
                between two sets of points set1 and set2.
           """
        # Calculate pairwise Euclidean distances between the points
        dist_matrix = np.sqrt(((set1[:, np.newaxis, :] - set2) ** 2).sum(axis=2))

        # Create a DataFrame of the distance matrix with formatting
        dist_df = pd.DataFrame(dist_matrix, index=[f"{i}'" for i in range(set1.shape[0])],
                               columns=[f"{j}'" for j in range(set2.shape[0])]).applymap("{:.3f}".format)

        # Find perfect match using the Hungarian algorithm.
        # The algorithm has a time complexity of O(n^3).
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        # ----------------------------------- TABLE DEFINITIONS ---------------------------------------------

        # # Create a table plot of the distance matrix with the matched distances highlighted
        # fig, ax = plt.subplots()
        # ax.axis('off')
        # table = ax.table(cellText=dist_df.values, rowLabels=dist_df.index, colLabels=dist_df.columns,
        #                  cellLoc='center', loc='center')
        #
        # # Color the matched distances in the table
        # for i, j in zip(row_ind, col_ind):
        #     table[i+1, j].set_text_props(weight='bold', color='blue')
        #
        # # Set the cell borders to be invisible
        # for key, cell in table.get_celld().items():
        #     cell.set_linewidth(0)
        #
        # # Resize the cells to fit the content
        # table.auto_set_font_size(False)
        # table.set_fontsize(8)
        # table.scale(1, 1.5)
        # ----------------------------- TERMINAL MATCHED POINTS DATA --------------------------

        # Plot matched points with labels and distances - terminal
        print("\nMatched points:")
        matches = []
        for i, j in zip(row_ind, col_ind):
            dist = dist_matrix[i][j]
            matches.append((set1[i], set2[j]))
            print(f"{i:3}' ({set1[i]} )  -> {j:3}' ({set2[j]} ), distance: {dist:>6.4f}")

        # Get the sum of distances in the optimal match
        total_sum = 0
        for i, j in zip(row_ind, col_ind):
            total_sum += dist_matrix[i][j]

        print("The sum of distances in optimal match is: ")
        print(total_sum)

        # ------------------------------------ BIPARTITE GRAPH PLOT ------------------------------------------
        # Generate the numerated sets according to the matching
        numerated_set1 = {f'{i}\'': point for i, point in enumerate(set1)}
        numerated_set2 = {f'{j}\'': set2[col_ind[i]] for i, j in enumerate(row_ind)}

        # Plot bipartite graph with edges labeled with distances
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot set 1 points in red
        ax.scatter(set1[:, 0], set1[:, 1], set1[:, 2], color='red', s=50, edgecolor='black', linewidths=2)

        # Plot set 2 points in blue
        ax.scatter(set2[:, 0], set2[:, 1], set2[:, 2], color='blue', s=50, edgecolor='black', linewidths=2)

        # Set axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('The perfect matches between the red set and  the blue set')

        # Add edges between matched points
        matched_indices = []
        labels = []  # List of labels for each matched pair of points
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]  # List of colors to cycle through

        for i, j in zip(row_ind, col_ind):
            x = [set1[i][0], set2[j][0]]
            y = [set1[i][1], set2[j][1]]
            z = [set1[i][2], set2[j][2]]
            dist = dist_matrix[i][j]
            edge_color = colors[len(matched_indices) % len(colors)]  # choose a color from the list of colors
            ax.plot(x, y, z, color=edge_color)
            matched_indices.append((i, j))
            labels.append(f"({i}',{j}')")  # Add label for matched pair

        # Add labels to the points from set 1
        for i, point in enumerate(set1):
            matched = False
            for idx, (i_matched, j_matched) in enumerate(matched_indices):
                if i == i_matched:
                    matched = True
                    j = j_matched
                    dist = dist_matrix[i][j]
                    ax.plot([point[0], set2[j][0]], [point[1], set2[j][1]], [point[2], set2[j][2]], color=f"C{idx}")
                    break
            if not matched:
                ax.text(point[0], point[1] + 0.2, point[2], f"{i}'", color='red', fontsize=13, weight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', pad=1))
            else:
                ax.text(point[0], point[1] + 0.2, point[2], f"{i}'", color=f"C{idx}", fontsize=13, weight='bold',
                        bbox=dict(facecolor='white', edgecolor=f"C{idx}", pad=1))

        # Add labels to the points from set 2
        for j, point in enumerate(set2):
            matched = False
            for idx, (i_matched, j_matched) in enumerate(matched_indices):
                if j == j_matched:
                    matched = True
                    i = i_matched
                    break
            if not matched:
                ax.text(point[0], point[1] + 0.3, point[2], f"{j}'", color='blue', fontsize=13, weight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', pad=1))
            else:
                ax.text(point[0], point[1] + 0.3, point[2], f"{j}'", color=f"C{idx}", fontsize=13, weight='bold',
                        bbox=dict(facecolor='white', edgecolor=f"C{idx}", pad=1))

        # Add legend
        legend_labels = []
        for idx, label in enumerate(labels):
            color = colors[idx % len(colors)]
            legend_labels.append(ax.plot([], [], color=color, label=label)[0])

        ax.legend(handles=legend_labels, loc='lower right', bbox_to_anchor=(0.05, 0.05))

        # ----------------------------- MOVE THE PLOT WITH MOUSE -------------------------------------------
        def on_mouse_move(event):
            if event.button == 'down':
                ax.view_init(elev=ax.elev + (event.ydata - y0),
                             azim=ax.azim - (event.xdata - x0))
                fig.canvas.draw()

        fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
        x0, y0 = ax.azim, ax.elev

        plt.show()

        # Return numerated sets
        return numerated_set1, numerated_set2, matches

# ======================================================================================================

    def append_zero(self, lst):
        """
        Takes a list of 2d tuples, each containing two float values, and
        returns a new list where each tuple has been transformed by appending
        a 0 to its coordinates.
        """
        result = []
        for point in lst:
            new_point = point + (0,)
            result.append(new_point)
        return result

# ======================================================================================================

    def extract_sift_descriptors(self, image):
        """
          The function extracts the descriptors and the key-points from the input image.

          Inputs:
          - image: A numpy array representing the input image.

          Outputs:
          - descriptors: A numpy array of SIFT descriptors extracted from the input image.
          - key-points: A list of key points detected in the input image using the SIFT algorithm.
          """
        # Convert the input image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()

        # Detect key points and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # Return the descriptors and key points
        return descriptors, keypoints

# ======================================================================================================

    def detect_and_describe_keypoints(self, image):
        """
        Detect keypoints in each image and compute their descriptors using the specified feature detector.

        Inputs: image to extract the descriptors and the keypoints from.
        Outputs: descriptors_list, keypoints_list (lists of keypoints and descriptors for each image)
        """
        # Extract SIFT descriptors and key points
        descriptors, keypoints = self.extract_sift_descriptors(image)
        return descriptors, keypoints

# ======================================================================================================

    def detect_and_describe_keypoints_aux(self):
        """
       Aux function for detect_and_describe_keypoints(self, image), extracts the descriptors and the key-points from all
        the images in the specified directory.

        Inputs: None ( using self.images)
        Outputs: all descriptors_list, all keypoints_list (two lists of the keypoints and descriptors, each for
         all the images in the directory)
        """
        # Initialize lists to store descriptors and keypoints
        all_descriptors = []
        all_keypoints = []

        # Loop over each image in the list
        for image in self.images:
            # Extract SIFT descriptors and key points
            descriptors, keypoints = self.detect_and_describe_keypoints(image)  # Returns  (descriptors, keypoints) from one image

            # Add descriptors and keypoints to the lists
            all_descriptors.append(descriptors)
            all_keypoints.append(keypoints)

            # Print the total number of descriptors and keypoints extracted
            print('Total number of descriptors:', sum([d.shape[0] for d in all_descriptors]))
            print('Total number of keypoints:', sum([len(k) for k in all_keypoints]))

        return all_descriptors, all_keypoints

# ======================================================================================================

    def aux_run(self):
        """
         This function finds perfect matches between keypoints between each pair of frames in the list of images .

         Inputs: None
         Outputs:
         - Allmatches: A list of matches between keypoints in two sets of images.
          Example: Matches between frame 1 and frame 2 are Allmatches[0].


         Input: None

         Output:
         - Allmatches: A list of matches between each following frames as in example above.

         """
        setA = []
        setB = []
        Allmatches = []

        all_descriptors, all_keypoints = self.detect_and_describe_keypoints_aux()
        for j, keypoint in enumerate(all_keypoints):
            setA.append(keypoint[j].pt)
            setB.append(keypoint[j + 1].pt)

        for i in range(0, len(all_keypoints) - 1):
            keypointsA = all_keypoints[i]
            keypointsB = all_keypoints[i + 1]
            setA = [kp.pt for kp in keypointsA]
            setB = [kp.pt for kp in keypointsB]

            setA = self.append_zero(setA)
            setB = self.append_zero(setB)

            setA = np.array(setA)
            setB = np.array(setB)


            matchA, matchB, matches = self.find_perfect_match(setA, setB)

            Allmatches.append(matches)

        return Allmatches, all_descriptors, all_keypoints

# ======================================================================================================

    def match_keypoints(self):
        """
        Match keypoints across pairs of images using their descriptors.

        Inputs: keypoints list.
        Outputs: matches_list (list of matches for each pair of images)
        """
        print("Matching key-points...")
        allTheMatches,  all_descriptors, all_keypoints = self.aux_run()

        return allTheMatches,  all_descriptors, all_keypoints

# ======================================================================================================

    def robustly_estimate_matches(self, matches_list):
        """
        Apply a robust estimation technique (e.g., RANSAC) to remove outlier matches.

        Inputs: matches_list (list of matches for each pair of images)
        Outputs: filtered_matches_list (list of filtered matches for each pair of images)
        """
        pass

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
        Execute the System pipeline by incrementally reconstructing the 3D structure.
        Inputs: None (uses self.images and self.feature_detector)
        Outputs: final_structure (final 3D point cloud)
        """

        # Step 1: Detect keypoints, match across pairs of images, estimate matches.
        # Returns :  list of all the matches. Match 1 (between two first frames) = allTheMatches[0]
        allTheMatches,  all_descriptors, all_keypoints = self.match_keypoints()

        print("Done matching!")

        # Step 3: Dense reconstruction:
        # =========================== START ADDING HERE =======================


        # # Step 3.a: Initialize the 3D structure and camera poses
        # initial_structure, initial_camera_poses = self.initialize_structure(keypoints, filtered_matches_list, )
        #
        # # Initialize the final structure and camera poses with the initial values
        # final_structure = initial_structure
        # final_camera_poses = initial_camera_poses
        #
        # # Iterate through the remaining images
        # for i in range(len(self.images)):
        #     if i not in initial_camera_poses:
        #         # Step 5a: Estimate the camera pose for the new image
        #         camera_pose = self.estimate_camera_pose(keypoints_list[i], filtered_matches_list[i], final_structure)
        #
        #         # Step 5b: Triangulate new points
        #         new_points = self.triangulate_new_points(
        #             keypoints_list[i], keypoints_list[final_camera_poses.keys()[0]],
        #             camera_pose, final_camera_poses[final_camera_poses.keys()[0]], filtered_matches_list[i]
        #         )
        #
        #         # Step 5c: Update existing points
        #         updated_structure = self.update_existing_points(
        #             keypoints_list[i], keypoints_list[final_camera_poses.keys()[0]],
        #             filtered_matches_list[i], final_structure
        #         )
        #
        #         # Step 5d: Bundle adjustment
        #         refined_camera_poses, refined_structure = self.bundle_adjustment(
        #             final_camera_poses, updated_structure, keypoints_list, filtered_matches_list
        #         )
        #
        #         # Update the final structure and camera poses with the refined values
        #         final_structure = refined_structure
        #         final_camera_poses = refined_camera_poses
        #
        # # Step 6: Detect loop closures and apply global optimization (optional)
        # loop_closures = self.detect_loop_closure(keypoints_list, descriptors_list, final_camera_poses)
        # optimized_camera_poses, optimized_structure = self.global_optimization(
        #     final_camera_poses, final_structure, loop_closures)
        #
        # # Update the final structure and camera poses with the optimized values
        # final_structure = optimized_structure
        # final_camera_poses = optimized_camera_poses
        #
        # return final_camera_poses, final_structure

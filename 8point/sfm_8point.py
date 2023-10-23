import numpy as np
import cv2
import matplotlib.pyplot as plt

class StructureFromMotion:
    def __init__(self, images, feature_detector='SIFT'):
        self.images = images
        self.feature_detector = feature_detector

    def best_R_T(self, points1, points2, R, T, K):
        min_error = float('inf')
        best_R = None
        best_T = None
        for i in range(len(R)):
            P1 = K @ np.eye(3, 4)
            P2 = K @ np.hstack((R[i], T[i][:, np.newaxis]))
            error = 0
            for j in range(len(points1)):
                x1_hom = np.array([points1[j][0], points1[j][1], 1, 0])
                x2_hom = np.array([points2[j][0], points2[j][1], 1, 0])
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
        if K is None:
            K = np.eye(3)
        E, _ = cv2.findEssentialMat(points1, points2, K, method=cv2.FM_8POINT, threshold=0.1)
        _, R, t, _ = cv2.recoverPose(E, points1, points2, K)
        return R, t

    def use8point(self, points1, points2, K=None):
        if K is None:
            K = np.eye(3)
        x1 = points1[:, 0]
        y1 = points1[:, 1]
        x2 = points2[:, 0]
        y2 = points2[:, 1]
        A = np.zeros((len(x1), 9))
        for i in range(len(x1)):
            A[i] = [x2[i] * x1[i], x2[i] * y1[i], x2[i], y2[i] * x1[i], y2[i] * y1[i], y2[i], x1[i], y1[i], 1]
        U, S, V = np.linalg.svd(A, full_matrices=True)
        F_est = V[-1, :].reshape(3, 3)
        ua, sa, va = np.linalg.svd(F_est, full_matrices=True)
        sa = np.diag(sa)
        sa[2, 2] = 0
        F = np.dot(ua, np.dot(sa, va))
        E_est = np.dot(K.T, np.dot(F, K))
        U, S, V = np.linalg.svd(E_est, full_matrices=True)
        S = np.diag([1, 1, 0])
        E = np.dot(U, np.dot(S, V))
        U, S, V = np.linalg.svd(E)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
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
            if np.linalg.det(R[i]) < 0:
                R[i] = -R[i]
                T[i] = -T[i]
        R, T = self.best_R_T(points1, points2, R, T, K)
        return R, T

def compute_reprojection_error(points1, points2, R, T, K):
    P1 = K @ np.eye(3, 4)
    P2 = K @ np.hstack((R, T.reshape(-1, 1)))
    total_error = 0
    for i in range(len(points1)):
        X_hom = threeD_points_homogeneous[i]
        x1_reproj = P1 @ X_hom
        x2_reproj = P2 @ X_hom
        x1_reproj = x1_reproj[:2] / x1_reproj[2]
        x2_reproj = x2_reproj[:2] / x2_reproj[2]
        error = np.linalg.norm(x1_reproj - points1[i]) + np.linalg.norm(x2_reproj - points2[i])
        total_error += error
    avg_error = total_error / len(points1)
    return avg_error

# Generate random 3D structure with at least 10 points
num_points = 10
threeD_points = np.random.rand(num_points, 3) * 10

# Define fixed R, T
angle = np.pi / 6  # 30 degrees
R_fixed = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1]
])
T_fixed = np.array([5, 0, 0])  # Translate 5 units along x-axis

# Create two reprojections using fixed R and T
K = np.array([[800, 0, 500], [0, 800, 500], [0, 0, 1]])
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K @ np.hstack((R_fixed, T_fixed[:, np.newaxis]))


# Initialize the StructureFromMotion instance
sfm = StructureFromMotion(images=[])

# Convert 3D points to homogeneous coordinates
threeD_points_homogeneous = np.hstack((threeD_points, np.ones((num_points, 1))))

# Recompute the reprojections using fixed R and T
points1 = (P1 @ threeD_points_homogeneous.T).T
points1 /= points1[:, 2][:, np.newaxis]
points1 = points1[:, :2]

points2 = (P2 @ threeD_points_homogeneous.T).T
points2 /= points2[:, 2][:, np.newaxis]
points2 = points2[:, :2]

# Using 8-point OpenCV
R_opencv, T_opencv = sfm.use8point_opencv(points1, points2, K)
T_opencv = T_opencv.reshape(-1)
# Using 8-point
R_manual, T_manual = sfm.use8point(points1, points2, K)

# Errors between computed and fixed R, T (Before Rescale)
error_R_opencv_before = np.linalg.norm(R_opencv - R_fixed)
error_T_opencv_before = np.linalg.norm(T_opencv - T_fixed)
error_R_manual_before = np.linalg.norm(R_manual - R_fixed)
error_T_manual_before = np.linalg.norm(T_manual - T_fixed)

# Compute the reprojection error for both methods
error_opencv_reprojection = compute_reprojection_error(points1, points2, R_opencv, T_opencv, K)
error_manual_reprojection = compute_reprojection_error(points1, points2, R_manual, T_manual, K)

print("Errors (Before Rescale):")
print("OpenCV: R:", error_R_opencv_before, "T:", error_T_opencv_before)
print("Manual: R:", error_R_manual_before, "T:", error_T_manual_before)
print("\nReprojection Errors:")
print("OpenCV:", error_opencv_reprojection)
print("Manual:", error_manual_reprojection)

# Rescale the translations
magnitude_T_fixed = np.linalg.norm(T_fixed)

# Normalize and rescale
T_opencv_rescaled = (T_opencv / np.linalg.norm(T_opencv)) * magnitude_T_fixed
T_manual_rescaled = (T_manual / np.linalg.norm(T_manual)) * magnitude_T_fixed

# Errors between computed and fixed R, T (After Rescale)
error_R_opencv_after = error_R_opencv_before  # Rotation remains unchanged
error_T_opencv_after = np.linalg.norm(T_fixed - abs(T_opencv_rescaled))
error_R_manual_after = error_R_manual_before  # Rotation remains unchanged
error_T_manual_after = np.linalg.norm(T_fixed - abs(T_manual_rescaled))

print("\nErrors (After Rescale):")
print("OpenCV: R:", error_R_opencv_after, "T:", error_T_opencv_after)
print("Manual: R:", error_R_manual_after, "T:", error_T_manual_after)


# Compute the reprojection error for both methods
error_opencv_reprojection = compute_reprojection_error(points1, points2, R_opencv, abs(T_opencv_rescaled), K)
error_manual_reprojection = compute_reprojection_error(points1, points2, R_manual, abs(T_manual_rescaled), K)

print("\nReprojection Errors:")
print("OpenCV:", error_opencv_reprojection)
print("Manual:", error_manual_reprojection)

# Plotting the 3D points and their 2D projections
fig = plt.figure(figsize=(15, 5))

# 3D plot
ax = fig.add_subplot(131, projection='3d')
ax.scatter(threeD_points[:, 0], threeD_points[:, 1], threeD_points[:, 2], color='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Points')

# Projection 1
ax2 = fig.add_subplot(132)
ax2.scatter(points1[:, 0], points1[:, 1], color='r', marker='x')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
title_text = f"Projection 1"
ax2.set_title(title_text)

# Projection 2
ax3 = fig.add_subplot(133)
ax3.scatter(points2[:, 0], points2[:, 1], color='g', marker='s')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
title_text = f"Projection 2 (Transformed)"
ax3.set_title(title_text)

plt.tight_layout()
plt.show()
from pipeline import StructureFromMotion
import os
import cv2


def video_to_frames(video_path, output_dir):
    """
      Extracts frames from a video file and saves them as individual images.

      Inputs:
      - video_path: a string containing the path to the video file
      - output_dir: a string containing the path to the output directory

      Outputs:
      - None
      """

    # Load the video
    video = cv2.VideoCapture(video_path)

    # Get the frame count and fps of the video
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Calculate the step value to get 30 frames evenly spaced across the video
    step = int(frame_count / 80)

    # Loop through the frames and save every nth frame as an image
    for i in range(0, frame_count, step):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if not ret:
            break
        cv2.imwrite(f"{output_dir}/frame{i}.jpg", frame)

    # Release the video object
    video.release()

def read_images_from_directory(directory):
    """
    Reads all images from a directory and returns them as a list of NumPy arrays.

    Inputs:
    - directory: a string containing the path to the directory

    Outputs:
    - images: a list of NumPy arrays, where each array contains the pixel values of one image
    """
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            path = os.path.join(directory, filename)
            image = cv2.imread(path)
            images.append(image)
    return images


if __name__ == '__main__':
    directory = "./input"
    #video_to_frames(directory, directory)
    images = read_images_from_directory(directory)  # Load your images here as a list of NumPy arrays
    sfm = StructureFromMotion(images=images)
    sfm.run()

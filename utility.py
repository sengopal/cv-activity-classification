import os

import cv2
import numpy
import scipy
import setuptools
import sklearn
import platform
import h5py
import cython
# import pytorch
import torchvision
import skorch
import matplotlib

MY_VID_DIR = "my_videos"

## VIDEO Utilities - Copied from PS3
def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None


# Utility to save the video frames in the 'sample' folder for introspection
def save_video_to_images(file_name):
    image_gen = video_frame_generator(os.path.join(MY_VID_DIR, file_name + ".avi"))

    curr_image = image_gen.__next__()
    frame_num = 1
    while curr_image is not None:
        cv2.imwrite(os.path.join("sample", file_name + '_' + str(frame_num) + '.png'), curr_image)
        frame_num += 1
        curr_image = image_gen.__next__()


def version_check():
    # name: cv_proj
    # channels:
    #   - pytorch
    #   - anaconda
    #   - defaults
    # dependencies:
    #   - h5py=2.9.0
    #   - numpy=1.15.3
    #   - python=3.6.7
    #   - scipy=1.2.1
    #   - setuptools=39.1.0
    #   - cython=0.29
    #   - pytorch=1.0.1
    #   - torchvision=0.2.1
    #   - pip:
    #     - scikit-learn==0.20.0
    #     - opencv-python==3.4.3.18
    #     - skorch==0.5.0.post0

    assert (h5py.__version__ == '2.9.0')
    assert (numpy.__version__ == '1.15.3')
    assert (platform.python_version() == '3.6.7')
    assert (scipy.__version__ == '1.2.1')
    assert (setuptools.__version__ == '39.1.0')

    assert (cython.__version__.startswith('0.29'))  # 0.29.7
    # assert (pytorch.__version__ == '1.0.1')
    assert (torchvision.__version__ == '0.2.1')

    assert (sklearn.__version__ == '0.20.0')

    assert (cv2.__version__ == '3.4.3') # Version does not return 3.4.3.18
    assert (skorch.__version__ == '0.5.0.post0')
    assert (matplotlib.__version__ == '3.0.3')

if __name__ == '__main__':
    # save_video_to_images("sample08_multi_action_uncomp")
    version_check()

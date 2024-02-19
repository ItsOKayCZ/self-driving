import numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

THRESHOLD = 140


def img_threshold(img: numpy.ndarray) -> numpy.ndarray:
    return (img >= THRESHOLD).astype("float32")


def rgb_to_gray(rgb: numpy.ndarray) -> numpy.ndarray:
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def img_process(img: np.ndarray):
    img = rgb_to_gray(img)
    # img = img_threshold(img)
    return img


if __name__ == "__main__":
    img = cv2.imread("../../aws-ros2/test.png")
    img1 = cv2.imread("../../aws-ros2/test1.png")
    img_new = img_process(img)
    plt.imshow(img_new, cmap=plt.get_cmap('gray'))
    plt.show()
import cv2


def save_image(path, image):
    cv2.imwrite(path, image)

import cv2

class ImageLoader:
    def load_grayscale(self, path: str):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def load_color(self, path: str):
        return cv2.imread(path)
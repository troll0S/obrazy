import cv2
import numpy as np
from PIL import Image

class ImageManager:
    def __init__(self):
        self.original = None
        self.current = None
        self._is_grayscale = False

    def load_image(self, path):
        self.original = cv2.imread(path)
        self.current = self.original.copy()
        self._is_grayscale = self._detect_grayscale(self.current)

    def get_display_image(self):
        if self.current is None:
            return None
        bgr = cv2.cvtColor(self.current, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(bgr)
        return pil_img

    def save_image(self, path):
        if self.current is None:
            return False
        return cv2.imwrite(path, self.current)

    def resize_current(self, scale: float):
        if self.original is None:
            return
        height, width = self.original.shape[:2]
        new_size = (int(width*scale),int(height*scale))
        self.current = cv2.resize(self.original,new_size,interpolation=cv2.INTER_LINEAR)

    def _detect_grayscale(self, img, tolerance=0):
        if self.current is None:
            return False
        if len(self.current.shape) < 3 or self.current.shape[2] == 1:
            return True
        b,g,r = cv2.split(self.current)
        diff1 = cv2.absdiff(b,g)
        diff2 = cv2.absdiff(b,r)

        return np.max(diff1) <= tolerance and np.max(diff2) <= tolerance

    def is_grayscale(self):
        return self._is_grayscale
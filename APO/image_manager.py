import cv2
from PIL import Image

class ImageManager:
    def __init__(self):
        self.original = None
        self.current = None

    def load_image(self, path):
        self.original = cv2.imread(path)
        self.current = self.original.copy()

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

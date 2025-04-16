import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class ImageManager:
    def __init__(self):
        self.original = None
        self.current = None
        self._is_grayscale = False
        self.histogram_shown = False
        self.hist_window = None
        self.hist_fig = None

    def load_image(self, path):
        self.original = cv2.imread(path)
        self.current = self.original.copy()
        self._is_grayscale = self._detect_grayscale(self.current)

    def get_display_image(self):
        if self.current is None:
            return None

        if len(self.current.shape) == 2:  # Obraz jest w skali szarości
            pil_img = Image.fromarray(self.current)
        else:
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
        new_size = (int(width * scale), int(height * scale))
        self.current = cv2.resize(self.current, new_size, interpolation=cv2.INTER_LINEAR)

    def _detect_grayscale(self, img, tolerance=0):
        if self.current is None:
            return False
        if len(self.current.shape) < 3 or self.current.shape[2] == 1:
            return True
        b, g, r = cv2.split(self.current)
        diff1 = cv2.absdiff(b, g)
        diff2 = cv2.absdiff(b, r)

        return np.max(diff1) <= tolerance and np.max(diff2) <= tolerance

    def is_grayscale(self):
        return self._is_grayscale

    def rgb_to_gray(self):
        if self.current is None:
            return
        if self._is_grayscale:
            return
        gray = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        self.current = gray
        self._is_grayscale = True

    def split_rgb_to_grays(self):
        if self.current is None or self.is_grayscale() or self.is_rgb():
            return None
        b, g, r = cv2.split(self.current)
        return [b, g, r]

    def rgb_to_HSV(self):
        if self.current is None or self.is_grayscale() or self.is_rgb():
            return
        hsv = cv2.cvtColor(self.current, cv2.COLOR_RGB2HSV)
        self.current = hsv

    def rgb_to_lab(self):
        if self.current is None or self.is_grayscale() or self.is_rgb():
            return
        lab = cv2.cvtColor(self.current, cv2.COLOR_RGB2LAB)
        self.current = lab

    def is_rgb(self):
        if self.current is None:
            return False
        return (len(self.current.shape) == 3 and self.current.shape[2] == 3)

    def draw_histogram(self):
        if self.current is None or not self._is_grayscale:
            return
        if self.hist_window:
            plt.close(self.hist_fig)

        self.hist_window = plt.figure(f"Histogram_{id(self)}")
        self.hist_fig = self.hist_window

        hist, bins = np.histogram(self.current.flatten(), bins=256, range=(0, 256))
        max_val = hist.max()

        plt.title("Histogram obrazu szaroodcieniowego")
        plt.xlabel("Wartość piksela")
        plt.ylabel("Liczba pikseli")
        plt.ylim(0, max_val * 1.05)  # dynamiczne skalowanie
        plt.plot(bins[:-1], hist, color='gray')
        plt.grid(True)

        self.histogram_shown = True

        def on_close(event):
            self.histogram_shown = False
            self.hist_window = None
            self.hist_fig = None

        self.hist_fig.canvas.mpl_connect('close_event', on_close)
        plt.show(block=False)

    def show_LUT_table(self):
        pass

    def normalize(self):
        pass

    def equalize(self):
        pass

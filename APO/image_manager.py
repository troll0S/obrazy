import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

class ImageManager:
    def __init__(self):
        self.original = None
        self.current = None
        self._is_grayscale = False
        self.histogram_shown = False
        self.hist_window = None
        self.hist_fig = None
        self.filename = None
        self.lut_table = None
        self.lut_window = None

    def load_image(self, path):
        self.original = cv2.imread(path)
        self.current = self.original.copy()
        self._is_grayscale = self._detect_grayscale(self.current)
        self.filename = Path(path).stem

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

        self.hist_window = plt.figure(f"Histogram_{self.filename}_{id(self)}")
        self.hist_fig = self.hist_window

        hist, bins = np.histogram(self.current.flatten(), bins=256, range=(0, 256))
        max_val = hist.max()

        plt.xlabel("Wartość piksela")
        plt.ylim(0, max_val)
        plt.bar(bins[:-1], hist, width =1, color='gray')
        plt.grid(True)

        self.histogram_shown = True

        def on_close(event):
            self.histogram_shown = False
            self.hist_window = None
            self.hist_fig = None

        self.hist_fig.canvas.mpl_connect('close_event', on_close)
        plt.show(block=False)

    def get_histogram_data(self):
        if self.current is None or not self._is_grayscale:
            return None, None
        hist, bins = np.histogram(self.current.flatten(), bins=256, range=(0, 256))
        return hist, bins

    def show_lut_table(self):
        self.calc_lut()
        if self.lut_window is not None and self.lut_window.winfo_exists():
            self.lut_window.destroy()
            self.lut_table = None
            self.calc_lut()
        return self.lut_table


    def normalize(self):
        if self.current is None or not self._is_grayscale:
            return
        self.calc_lut()
        min_val = np.min(np.where(self.lut_table[1] > 0))
        max_val = np.max(np.where(self.lut_table[1] > 0))
        if min_val == max_val:
            return

        self.lut = np.zeros(256,dtype=np.uint8)
        for i in range(256):
            if i < min_val:
                self.lut[i] = 0
            elif i > max_val:
                self.lut[i] = 255
            else:
                self.lut[i] = ((i - min_val) * 255) // (max_val - min_val)

        self.apply_lut(self.lut)
        self.calc_lut()


    def equalize(self):
        if self.current is None or not self._is_grayscale:
            return
        self.calc_lut()
        total_pixels = self.current.size
        pdf = self.lut_table[1] / total_pixels
        cdf = np.cumsum(pdf)
        lut = np.round(cdf * 255).astype(np.uint8)
        self.apply_lut(lut)
        self.calc_lut()

    def negate(self):
        lut = np.array([255 - i for i in range(256)], dtype=np.uint8)
        self.apply_lut(lut)
        self.calc_lut()

    def posterize(self, levels=4):
        if self.current is None or not self._is_grayscale:
            return
        if levels < 2 or levels > 254:
            return
        max_val = 255
        min_val = 0
        step = (max_val+1)//levels
        indices = np.floor(self.current / step)
        centers = (indices + 0.5) * step
        self.current = np.clip(centers, min_val, max_val).astype(np.uint8)


    def apply_blur(self, border_mode):
        if self.current is None:
            return
        self.current = cv2.blur(self.current, (3,3),border_mode)

    def apply_gaussian_blur(self,border_mode):
        if self.current is None:
            return
        self.current = cv2.GaussianBlur(self.current, (3, 3), 0, borderType=border_mode)

    def apply_sobel(self, border_mode):
        if self.current is None:
            return
        sobel_x = cv2.Sobel(self.current, cv2.CV_64F, 1, 0, ksize=3, borderType=border_mode)
        sobel_y = cv2.Sobel(self.current, cv2.CV_64F, 0, 1, ksize=3, borderType=border_mode)

        sobel = cv2.magnitude(sobel_x, sobel_y)
        self.current = cv2.convertScaleAbs(sobel)

    def apply_laplacian(self,border_mode):
        if self.current is None:
            return
        laplacian = cv2.Laplacian(self.current, ddepth=cv2.CV_64F, ksize=3, borderType=border_mode)
        laplacian = cv2.convertScaleAbs(laplacian)
        self.current = laplacian

    def apply_canny(self, border_mode):
        if self.current is None:
            return
        sobel_x = cv2.Sobel(self.current, cv2.CV_64F, 1, 0, ksize=3, borderType=border_mode)
        sobel_y = cv2.Sobel(self.current, cv2.CV_64F, 0, 1, ksize=3, borderType=border_mode)
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        self.current = cv2.convertScaleAbs(sobel_combined)

    def apply_sharpen_laplace_cross(self,border_mode):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.apply_uniwersal(border_mode,kernel)

    def apply_sharpen_laplace_full(self,border_mode):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.apply_uniwersal(border_mode,kernel)

    def apply_sharpen_laplace_extreme(self,border_mode):
        kernel = np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])
        self.apply_uniwersal(border_mode,kernel)

    def apply_prewitt_n(self,border_mode):
        kernel = np.array([[ 1, 1, 1], [ 0, 0, 0], [-1, -1, -1]])
        self.apply_uniwersal(border_mode,kernel)

    def apply_prewitt_nw(self,border_mode):
        kernel = np.array([[ 1, 1, 0], [ 1, 0, -1], [ 0, -1, -1]])
        self.apply_uniwersal(border_mode,kernel)

    def apply_prewitt_w(self,border_mode):
        kernel = np.array([[ 1, 0, -1], [ 1, 0, -1], [ 1, 0, -1]])
        self.apply_uniwersal(border_mode,kernel)

    def apply_prewitt_sw(self,border_mode):
        kernel = np.array([[ 0, -1, -1], [ 1, 0, -1], [ 1, 1, 0]])
        self.apply_uniwersal(border_mode,kernel)

    def apply_prewitt_s(self,border_mode):
        kernel = np.array([[-1, -1, -1], [ 0, 0, 0], [ 1, 1, 1]])
        self.apply_uniwersal(border_mode,kernel)

    def apply_prewitt_se(self,border_mode):
        kernel = np.array([[-1, -1, 0], [-1, 0, 1], [ 0, 1, 1]])
        self.apply_uniwersal(border_mode,kernel)

    def apply_prewitt_e(self,border_mode):
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        self.apply_uniwersal(border_mode,kernel)

    def apply_prewitt_ne(self,border_mode):
        kernel = np.array([[ 0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
        self.apply_uniwersal(border_mode,kernel)

    def apply_uniwersal(self,border_mode,kernel):
        if self.current is None or kernel is None or border_mode is None:
            return
        kernel = np.array(kernel)
        self.current = cv2.filter2D(self.current, -1, kernel, borderType=border_mode)

    def apply_median_3x3(self,border_mode):
        if self.current is None or border_mode is None:
            return
        self.current = cv2.medianBlur(self.current, 3)

    def apply_median_5x5(self,border_mode):
        if self.current is None or border_mode is None:
            return
        self.current = cv2.medianBlur(self.current, 5)

    def apply_median_7x7(self,border_mode):
        if self.current is None or border_mode is None:
            return
        self.current = cv2.medianBlur(self.current, 7)



    def get_lut(self):
        return self.lut_table

    def is_lut_active(self):
        return self.lut_window is not None and self.lut_window.winfo_exists()

    def is_histogram_active(self):
        return self.histogram_shown is True and self.hist_window is not None and self.hist_fig is not None

    def apply_lut(self, lut):
        if self.current is None or not self._is_grayscale:
            return
        lut = np.array(lut, dtype=np.uint8)
        self.current = lut[self.current]

    def calc_lut(self):
        if self.current is None or not self._is_grayscale:
            return
        hist, _ = np.histogram(self.current.flatten(), bins=256, range=(0, 256))
        values = np.arange(256)
        self.lut_table = np.vstack((values, hist))


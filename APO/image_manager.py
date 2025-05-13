

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import math


class ImageManager:
    def __init__(self):
        self.original = None
        self.current = None
        self._is_grayscale = False
        self._is_binary = False
        self.histogram_shown = False
        self.hist_window = None
        self.hist_fig = None
        self.filename = None
        self.lut_table = None
        self.lut_window = None
        self.pyramid_images = None
        self.pyramid_active = None

    def load_image(self, path):
        self.original = cv2.imread(path)
        self.current = self.original.copy()
        self._is_grayscale = self._detect_grayscale(self.current)
        self._is_binary = self.is_binary(self.current)
        self.filename = Path(path).stem
        self.calc_pyramid()
        self.pyramid_active = 2

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

    def reset_image(self):
        if self.original is None:
            return
        self.current = self.original.copy()
        self._is_grayscale = self._detect_grayscale(self.current)
        self._is_binary = self.is_binary(self.current)

    def set_current(self,img):
        self.current = img
        self.calc_pyramid()
        _is_grayscale = self._detect_grayscale(self.current)
        _is_binary = self.is_binary(self.current)

    def _detect_grayscale(self, img, tolerance=0):
        if img is None:
            return False
        if len(img.shape) < 3 or img.shape[2] == 1:
            return True
        b, g, r = cv2.split(img)
        diff1 = cv2.absdiff(b, g)
        diff2 = cv2.absdiff(b, r)

        return np.max(diff1) <= tolerance and np.max(diff2) <= tolerance

    def is_grayscale(self):
        return self._is_grayscale

    def is_binary(self, img=None):
        if img is None:
            img = self.current
        if img is None:
            return False

        # Jeśli obraz ma jeden kanał — sprawdź wartości
        if img.ndim == 2:
            unique_values = set(np.unique(img))
            return unique_values.issubset({0, 255})

        # Jeśli obraz ma 3 kanały (np. RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            # Sprawdź, czy wszystkie kanały są identyczne
            if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 1], img[:, :, 2]):
                # Pobierz jeden kanał i sprawdź, czy ma tylko 0 i 255
                gray = img[:, :, 0]
                unique_values = set(np.unique(gray))
                if unique_values.issubset({0, 255}):
                    # Konwersja do jednego kanału
                    self.current = gray.copy()
                    return True

        return False

    def rgb_to_gray(self):
        if self.current is None:
            return
        if self._is_grayscale:
            return
        gray = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        self.set_current(gray)
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
        self.set_current(hsv)

    def rgb_to_lab(self):
        if self.current is None or self.is_grayscale() or self.is_rgb():
            return
        lab = cv2.cvtColor(self.current, cv2.COLOR_RGB2LAB)
        self.set_current(lab)

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
        plt.bar(bins[:-1], hist, width=1, color='gray')
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

    def normalize(self, target_min=0, target_max=255):
        if self.current is None or not self._is_grayscale:
            return
        self.calc_lut()
        min_val = np.min(np.where(self.lut_table[1] > 0))
        max_val = np.max(np.where(self.lut_table[1] > 0))
        if min_val == max_val:
            return

        self.lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            if i < min_val:
                self.lut[i] = target_min
            elif i > max_val:
                self.lut[i] = target_max
            else:
                self.lut[i] = ((i - min_val) * (target_max - target_min)) // (max_val - min_val) + target_min

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
        step = (max_val + 1) // levels
        indices = np.floor(self.current / step)
        centers = (indices + 0.5) * step
        self.set_current(np.clip(centers, min_val, max_val).astype(np.uint8))

    def apply_blur(self, border_mode):
        if self.current is None:
            return
        self.set_current(cv2.blur(self.current, (3, 3), border_mode))

    def apply_gaussian_blur(self, border_mode):
        if self.current is None:
            return
        self.set_current(cv2.GaussianBlur(self.current, (3, 3), 0, borderType=border_mode))

    def apply_sobel(self, border_mode):
        if self.current is None:
            return
        sobel_x = cv2.Sobel(self.current, cv2.CV_64F, 1, 0, ksize=3, borderType=border_mode)
        sobel_y = cv2.Sobel(self.current, cv2.CV_64F, 0, 1, ksize=3, borderType=border_mode)

        sobel = cv2.magnitude(sobel_x, sobel_y)
        self.set_current(cv2.convertScaleAbs(sobel))

    def apply_laplacian(self, border_mode):
        if self.current is None:
            return
        laplacian = cv2.Laplacian(self.current, ddepth=cv2.CV_64F, ksize=3, borderType=border_mode)
        laplacian = cv2.convertScaleAbs(laplacian)
        self.set_current(laplacian)

    def apply_canny(self, border_mode):
        if self.current is None:
            return
        sobel_x = cv2.Sobel(self.current, cv2.CV_64F, 1, 0, ksize=3, borderType=border_mode)
        sobel_y = cv2.Sobel(self.current, cv2.CV_64F, 0, 1, ksize=3, borderType=border_mode)
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        self.set_current(cv2.convertScaleAbs(sobel_combined))

    def apply_sharpen_laplace_cross(self, border_mode):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.apply_uniwersal(border_mode, kernel)

    def apply_sharpen_laplace_full(self, border_mode):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.apply_uniwersal(border_mode, kernel)

    def apply_sharpen_laplace_extreme(self, border_mode):
        kernel = np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])
        self.apply_uniwersal(border_mode, kernel)

    def apply_prewitt_n(self, border_mode):
        kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        self.apply_uniwersal(border_mode, kernel)

    def apply_prewitt_nw(self, border_mode):
        kernel = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])
        self.apply_uniwersal(border_mode, kernel)

    def apply_prewitt_w(self, border_mode):
        kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        self.apply_uniwersal(border_mode, kernel)

    def apply_prewitt_sw(self, border_mode):
        kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
        self.apply_uniwersal(border_mode, kernel)

    def apply_prewitt_s(self, border_mode):
        kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        self.apply_uniwersal(border_mode, kernel)

    def apply_prewitt_se(self, border_mode):
        kernel = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
        self.apply_uniwersal(border_mode, kernel)

    def apply_prewitt_e(self, border_mode):
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        self.apply_uniwersal(border_mode, kernel)

    def apply_prewitt_ne(self, border_mode):
        kernel = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
        self.apply_uniwersal(border_mode, kernel)

    def apply_uniwersal(self, border_mode, kernel):
        if self.current is None or kernel is None or border_mode is None:
            return
        kernel = np.array(kernel)
        self.set_current(cv2.filter2D(self.current, -1, kernel, borderType=border_mode))

    def apply_median_3x3(self, border_mode):
        if self.current is None or border_mode is None:
            return
        self.set_current(cv2.medianBlur(self.current, 3))

    def apply_median_5x5(self, border_mode):
        if self.current is None or border_mode is None:
            return
        self.set_current(cv2.medianBlur(self.current, 5))

    def apply_median_7x7(self, border_mode):
        if self.current is None or border_mode is None:
            return
        self.set_current(cv2.medianBlur(self.current, 7))

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
        self.set_current(lut[self.current])

    def calc_lut(self):
        if self.current is None or not self._is_grayscale:
            return
        hist, _ = np.histogram(self.current.flatten(), bins=256, range=(0, 256))
        values = np.arange(256)
        self.lut_table = np.vstack((values, hist))

    def apply_dual_image_operation(self, second_image_path, operation, alpha=0.5):
        if self.current is None or not self._detect_grayscale(self.current):
            return
        second_image = cv2.imread(second_image_path,cv2.IMREAD_GRAYSCALE)

        if second_image is None or not self._detect_grayscale(second_image):
            return
        if len(self.current.shape) == 3 and self.current.shape[2] == 3:
            self.set_current(cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY))
        second_image = cv2.resize(second_image, (self.current.shape[1], self.current.shape[0]))
        second_image = second_image.astype(self.current.dtype)

        print("Current shape:", self.current.shape, self.current.dtype)
        print("Second shape:", second_image.shape, second_image.dtype)

        if operation == 'add':
            result = cv2.add(self.current, second_image)
        elif operation == 'subtract':
            result = cv2.subtract(self.current, second_image)
        elif operation == 'blend':
            result = cv2.addWeighted(self.current, alpha, second_image, 1 - alpha, 0)
        elif operation == 'and':
            result = cv2.bitwise_and(self.current, second_image)
        elif operation == 'or':
            result = cv2.bitwise_or(self.current, second_image)
        elif operation == 'xor':
            result = cv2.bitwise_xor(self.current, second_image)
        else:
            return

        self.set_current(result)

    def apply_morphology(self,operation,shape,border,kernel_size):
        if self.current is None:
            return
        if self.is_binary(self.current) == False:
            return
        op_map = {
            "Erozja": cv2.MORPH_ERODE,
            "Dylacja": cv2.MORPH_DILATE,
            "Otwarcie": cv2.MORPH_OPEN,
            "Zamknięcie": cv2.MORPH_CLOSE
        }

        shape_map = {
            "Kwadrat": cv2.MORPH_RECT,
            "Romb": cv2.MORPH_CROSS
        }



        operation_code = op_map.get(operation)
        shape_code = shape_map.get(shape)
        border_code = border

        if operation_code is None or shape_code is None or border_code is None:
            return

        kernel = cv2.getStructuringElement(shape_code, (kernel_size, kernel_size))
        self.set_current(cv2.morphologyEx(self.current, operation_code, kernel, borderType=border_code))

    def skeletonize(self, border_mode):
        if self.current is None:
            raise ValueError("Brak obrazu.")
        if not self.is_binary():
            raise ValueError("Obraz nie jest binarny (musi zawierać tylko 0 i 255).")

        # Zamiana 255 -> 1
        image = (self.current.copy() // 255).astype(np.uint8)

        # Sąsiedzi w 8-kierunkowym sąsiedztwie
        neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                     (1, 0), (1, -1), (0, -1), (-1, -1)]

        # Wzorce bazowe
        base_patterns = [
            np.array([[0, 0, 0],
                      [-1, 1, -1],
                      [1, 1, 1]]),
            np.array([[-1, 0, 0],
                      [1, 1, 0],
                      [-1, 1, -1]])
        ]

        # Generowanie obrotów wzorców
        patterns = []
        for pat in base_patterns:
            for k in range(4):
                rotated = np.rot90(pat, k)
                if not any(np.array_equal(rotated, p) for p in patterns):
                    patterns.append(rotated)

        # Dopasowanie wzorca
        def match_pattern(region, pattern):
            mask = pattern != -1
            return np.array_equal(region[mask], pattern[mask])

        remain = True
        while remain:
            remain = False
            for j in [0, 2, 4, 6]:
                padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, border_mode, value=0)
                temp = padded.copy()
                rows, cols = padded.shape
                for x in range(1, rows - 1):
                    for y in range(1, cols - 1):
                        if padded[x, y] != 1:
                            continue
                        dx, dy = neighbors[j]
                        if padded[x + dx, y + dy] != 0:
                            continue

                        region = padded[x - 1:x + 2, y - 1:y + 2]

                        skel = False
                        for pattern in patterns:
                            if match_pattern(region, pattern):
                                skel = True
                                break

                        if skel:
                            temp[x, y] = 2  # zachowujemy jako szkielet
                        else:
                            temp[x, y] = 3  # usuwamy
                            remain = True

                # Aktualizacja: usuwanie i zachowywanie pikseli
                for x in range(1, rows - 1):
                    for y in range(1, cols - 1):
                        if temp[x, y] == 3:
                            temp[x, y] = 0
                        elif temp[x, y] == 2:
                            temp[x, y] = 1

                image = temp[1:-1, 1:-1].copy()

        self.set_current((image * 255).astype(np.uint8))


    def hough_transformation(self):
        if self.current is None:
            return
        if not self._detect_grayscale(self.current):
            return
        edges = cv2.Canny(self.current, 50, 200)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100, None, 0, 0)
        output = cv2.cvtColor(self.current,cv2.COLOR_GRAY2RGB)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

                cv2.line(output,pt1,pt2,(0,0,255),3,cv2.LINE_AA)
        self.set_current(output)
        self._is_grayscale = self._detect_grayscale(self.current)
        self._is_binary = self.is_binary(self.current)

    def calc_pyramid(self):
        self.pyramid_images = [None] * 5
        self.pyramid_images[2] = self.current.copy()  # oryginał na środku

        # Pomniejszenia
        self.pyramid_images[1] = cv2.pyrDown(self.pyramid_images[2])
        self.pyramid_images[0] = cv2.pyrDown(self.pyramid_images[1])

        # Powiększenia
        self.pyramid_images[3] = cv2.pyrUp(self.pyramid_images[2])
        self.pyramid_images[4] = cv2.pyrUp(self.pyramid_images[3])

    def pyramid_Up(self):
        if self.pyramid_active < 4:
            self.pyramid_active += 1
            self.current = self.pyramid_images[self.pyramid_active]
    def pyramid_Down(self):
        if self.pyramid_active > 0:
            self.pyramid_active -= 1
            self.current = self.pyramid_images[self.pyramid_active]
    def pyramid_Reset(self):
        self.pyramid_active = 2
        self.current = self.pyramid_images[self.pyramid_active]



    def threshold_manual(self, thresh_value):
        if self.current is None:
            raise ValueError("Brak obrazu.")

        if not self._detect_grayscale(self.current):
            return
        _, result_image = cv2.threshold(self.current, thresh_value, 255, cv2.THRESH_BINARY)
        self.set_current(result_image)

    def threshold_manual_reverse(self, thresh_value):
        if self.current is None:
            raise ValueError("Brak obrazu.")
        if not self._detect_grayscale(self.current):
            return
        _, result_image = cv2.threshold(self.current, thresh_value, 255, cv2.THRESH_BINARY_INV)
        self.set_current(result_image)

    def threshold_otsu(self):
        if self.current is None:
            raise ValueError("Brak obrazu.")
        if not self._detect_grayscale(self.current):
            return
        if self.is_binary(self.current):
            return
        if len(self.current.shape) == 3 and self.current.shape[2] == 3:
            self.set_current(cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY))
        _, result_image = cv2.threshold(self.current, 0, 255,
                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(self.is_binary(result_image))
        self.set_current(result_image)



    def threshold_adaptive(self, method, block_size=11, C=2):
        if self.current is None:
            raise ValueError("Brak obrazu.")
        if not self._detect_grayscale(self.current):
            return
        if len(self.current.shape) == 3 and self.current.shape[2] == 3:
            self.set_current(cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY))
        result_image = cv2.adaptiveThreshold(self.current, 255, method,
                                                  cv2.THRESH_BINARY, block_size, C)
        self.set_current(result_image)

    def grabcut(self, rect):
        if self.current is None:
            raise ValueError("Brak obrazu.")
        mask = np.zeros(self.current.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        x, y, w, h = rect
        cv2.grabCut(self.current, mask, (x, y, w, h), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = self.current * mask2[:, :, np.newaxis]
        self.set_current(result)

    def watershed(self):
        if self.current is None:
            raise ValueError("Brak obrazu.")
        if not self._detect_grayscale(self.current):
            return
        if len(self.current.shape) == 3 and self.current.shape[2] == 3:
            self.set_current(cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY))
        _, thresh = cv2.threshold(self.current, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown == 255] = 0
        img = self.current
        if len(img.shape) != 3 or img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        markers = np.int32(markers)
        markers = cv2.watershed(img,markers)
        img[markers == -1] = [255,0,0]
        self.set_current(img)


    def inpainting(self, mask_path):
        if self.current is None:
            return
        if not self._detect_grayscale(self.current):
            return
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None or not self._detect_grayscale(mask):
            return
        if len(self.current.shape) == 3 and self.current.shape[2] == 3:
            self.set_current(cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY))
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if mask.shape != self.current.shape:
            return
        result = cv2.inpaint(self.current, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        self.set_current(result)

    def analize_image(self):
        if self.current is None:
            return[]
        if not self.is_binary(self.current):
            return[]

        contours, _ = cv2.findContours(self.current, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for contour in contours:
            features = {}
            M = cv2.moments(contour)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = cy = 0

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            rect_area = w * h
            extent = float(area) / rect_area if rect_area != 0 else 0

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area != 0 else 0

            equi_diameter = np.sqrt(4 * area / np.pi)

            features.update({
                "cx": cx,
                "cy": cy,
                "m00": M["m00"],
                "area": area,
                "perimeter": perimeter,
                "aspect_ratio": aspect_ratio,
                "extent": extent,
                "solidity": solidity,
                "equivalent_diameter": equi_diameter
            })

            result.append(features)

        return result

    def rle_compress(self):
        if self.current is None or not self._is_grayscale:
            return None
        if len(self.current.shape) == 3 and self.current.shape[2] == 3:
            self.set_current(cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY))
        flat = self.current.flatten()
        compressed = []
        prev_pixel = flat[0]
        count = 1
        for pixel in flat[1:]:
            if pixel == prev_pixel:
                count += 1
            else:
                compressed.append((prev_pixel, count))
                prev_pixel = pixel
                count = 1
        compressed.append((prev_pixel, count))
        original_size = len(flat)
        compressed_size = len(compressed) * 2
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
        return {
            "compressed_data": compressed,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio
        }
import cv2

class ImageModifier:
    def adjust_brightness(self, image, alpha, beta):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def update_histogram(self, image, histogram):
        return histogram.create(image)
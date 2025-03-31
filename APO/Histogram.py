import numpy as np
import matplotlib.pyplot as plt

class Histogram:
    def create(self, image):
        hist = np.zeros(256)
        for pixel in image.flatten():
            hist[pixel] += 1
        return hist

    def display_graphical(self, hist):
        plt.plot(hist)
        plt.title('Histogram')
        plt.xlabel('Intensywność')
        plt.ylabel('Liczba pikseli')
        plt.show()

    def display_tabular(self, hist):
        print(hist)

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageLoader:
    def load_grayscale(self, path: str):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def load_color(self, path: str):
        return cv2.imread(path)

class ImageProcessor:
    def convert_to_gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def split_channels(self, image):
        return cv2.split(image)

    def convert_to_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def convert_to_lab(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

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

class ImageModifier:
    def adjust_brightness(self, image, alpha, beta):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def update_histogram(self, image, histogram):
        return histogram.create(image)

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikacja do Przetwarzania Obrazów")

        # Inicjalizacja klas
        self.image_loader = ImageLoader()
        self.image_processor = ImageProcessor()
        self.histogram = Histogram()
        self.image_modifier = ImageModifier()

        self.images = []  # Lista do przechowywania wielu obrazów
        self.image_labels = []  # Lista do przechowywania etykiet dla obrazów

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack()

        self.create_widgets()

    def create_widgets(self):
        # Przycisk do załadowania obrazu
        self.load_button = tk.Button(self.root, text="Wczytaj obraz", command=self.load_image)
        self.load_button.pack()

        # Przycisk do przetwarzania obrazu (generowanie histogramu)
        self.process_button = tk.Button(self.root, text="Przetwórz obraz", command=self.process_image)
        self.process_button.pack()

        # Przycisk do modyfikacji obrazu
        self.modify_button = tk.Button(self.root, text="Zmodyfikuj obraz", command=self.modify_image)
        self.modify_button.pack()

        # Przycisk do wyświetlania histogramu
        self.histogram_button = tk.Button(self.root, text="Pokaż histogram", command=self.show_histogram)
        self.histogram_button.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.bmp;*.jpg;*.png;*.gif")])
        if file_path:
            img = self.image_loader.load_color(file_path)
            self.images.append(img)
            self.display_images()

    def display_images(self):
        # Czyścimy poprzednie obrazy w GUI
        for label in self.image_labels:
            label.destroy()

        self.image_labels.clear()  # Resetujemy listę etykiet

        # Wyświetlanie wszystkich obrazów w self.images
        for img in self.images:
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)

            # Tworzymy etykietę i wyświetlamy obraz w oknie
            img_label = tk.Label(self.canvas_frame, image=image_tk)
            img_label.image = image_tk  # Przechowujemy referencję do obrazu
            img_label.pack(side=tk.LEFT)  # Wyświetlamy obrazy obok siebie
            self.image_labels.append(img_label)

    def process_image(self):
        if self.images:
            gray_image = self.image_processor.convert_to_gray(self.images[-1])
            self.images.append(gray_image)
            self.display_images()
            hist = self.histogram.create(gray_image)
            self.histogram.display_graphical(hist)
            self.histogram.display_tabular(hist)
        else:
            messagebox.showerror("Błąd", "Nie wczytano obrazu!")

    def modify_image(self):
        if self.images:
            modified_image = self.image_modifier.adjust_brightness(self.images[-1], alpha=1.2, beta=30)
            self.images.append(modified_image)
            self.display_images()
            hist = self.histogram.create(modified_image)
            self.histogram.display_graphical(hist)
            self.histogram.display_tabular(hist)
        else:
            messagebox.showerror("Błąd", "Nie wczytano obrazu!")

    def show_histogram(self):
        if self.images:
            gray_image = self.image_processor.convert_to_gray(self.images[-1])
            hist = self.histogram.create(gray_image)
            self.histogram.display_graphical(hist)
            self.histogram.display_tabular(hist)
        else:
            messagebox.showerror("Błąd", "Nie wczytano obrazu!")

# Uruchamianie aplikacji
if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()

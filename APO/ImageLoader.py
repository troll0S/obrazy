import cv2
import os
from tkinter import filedialog


class ImageLoader:
    def __init__(self):
        self.master = None
        self.images = []
    def load_images(self):

        file_paths = filedialog.askopenfilenames(
            title="Wybierz obrazy (można te same)",
            filetypes=[("Pliki graficzne", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )

        if file_paths:
            for i, path in enumerate(file_paths):
                try:
                    image = cv2.imread(path)
                    if image is None:
                        raise IOError("Nie udało się wczytać obrazu (możliwe ograniczenia dostępu)")
                    win_name = f"{os.path.basename(path)}"
                    cv2.imshow(win_name, image)
                    self.images.append(image)
                except Exception as e:
                    print(f"Błąd przy wczytywaniu pliku {path}: {e}")

            return self.images
        else:
            print("Nie wybrano plików.")
            return []

    def get_images(self):
        return self.images

if __name__ == "__main__":
    pass
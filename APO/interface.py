import tkinter as tk
from ImageLoader import ImageLoader
from tkinter import filedialog, messagebox
import cv2
class ImageLoaderApp:
    def __init__(self, master):
        self.master = master
        master.title("APO app")
        master.geometry("300x100")


        menubar = tk.Menu(master)
        master.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Plik", menu=file_menu)
        file_menu.add_command(label="Otwórz", command=self.load_images)
        file_menu.add_command(label="Duplikuj", command=self.duplicate_image)
        #file_menu.add_command(label="Zapisz", command=self.save_image) #Nie działa
        file_menu.add_command(label="RGB to Grey") #Dodać implementacje
        file_menu.add_command(label="Transform to mono chrom") #Dodać implementacje
        file_menu.add_command(label="Zamknij", command=self.close_app)
        file_menu.add_command(label="O autorze", command=self.show_about)

        histogram_menu = tk.Menu(menubar,tearoff=0)
        menubar.add_cascade(label="histogram", menu=histogram_menu)
        histogram_menu.add_command(label="Pokaż")
        histogram_menu.add_command(label="Pokaż tablicę LUT")
        histogram_menu.add_command(label="Rozciąganie")
        histogram_menu.add_command(label="Egualizacja")

        jednopunkt_menu = tk.Menu(menubar,tearoff=0)
        menubar.add_cascade(label="operacje jednopunktowe", menu=jednopunkt_menu)
        jednopunkt_menu.add_command(label="rozciąganie selektywne")
        jednopunkt_menu.add_command(label="negacja")


    def load_images(self):
        # Tworzymy instancję klasy ImageLoader
        loader = ImageLoader()
        loader.load_images()

    def save_image(self):
        if not self.loaded_images:
            messagebox.showwarning("Brak obrazu", "Nie wczytano obrazu do zapisania.")
            return

        # Okno dialogowe do wyboru lokalizacji zapisu
        file_path = filedialog.asksaveasfilename(
            title="Zapisz obraz",
            filetypes=[("Pliki graficzne", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")],
            defaultextension=".png"
        )

        if file_path:
            try:

                cv2.imwrite(file_path, self.loaded_images[0])
                messagebox.showinfo("Sukces", "Obraz został zapisany.")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się zapisać obrazu: {e}")
        else:
            print("Zapisanie obrazu zostało anulowane.")
    def duplicate_image(self):
       pass


    def close_app(self):
        self.master.quit()

    def show_about(self):
        tk.messagebox.showinfo("O autorze", "Autor: Jan Kowalski \nProwadzący: dr inż. Łukasz Roszkowiak\nAlgorytmy Przetwarzania Obrazów 2025 \nWIT grupa ID: IO1")


if __name__ == "__main__":
    pass

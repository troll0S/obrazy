import tkinter as tk
from ImageLoader import ImageLoader

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
        file_menu.add_command(label="Zamknij", command=self.close_app)
        file_menu.add_command(label="O autorze", command=self.show_about)


    def load_images(self):
        # Tworzymy instancję klasy ImageLoader
        loader = ImageLoader()
        loader.load_images()

    def duplicate_image(self):
       pass


    def close_app(self):
        self.master.quit()

    def show_about(self):
        tk.messagebox.showinfo("O autorze", "Aplikacja do wczytywania obrazów\nAutor: Twoje Imię")

if __name__ == "__main__":
    pass

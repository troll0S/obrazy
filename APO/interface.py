import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from image_manager import ImageManager
from tkinter import messagebox
import os


class Interface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplikacja do przetwarzania obrazów")
        self.geometry("300x150")

        self.create_menubar()

        self.windows = []
        self.active_window = None

    def create_menubar(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_command(label="Save Active Image", command=self.save_active_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        menubar.add_cascade(label="File", menu=file_menu)

        self.config(menu=menubar)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Obrazy", "*.jpg *.png *.bmp *.jpeg")]
        )
        if path:
            image_win = ImageWindow(self, path)
            self.windows.append(image_win)
            self.set_active_window(image_win)

    def set_active_window(self, window):
        self.active_window = window
        self.title(f"Menadżer obrazów - aktywne: {window.title()}")

    def display_image(self):
        img = self.image_manager.get_display_image()
        if img is not None:
            self.tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_img)

    def save_active_image(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if file_path:
            success = self.active_window.manager.save_image(file_path)
            if success:
                messagebox.showinfo("Sukces", f"Obraz zapisany jako: {os.path.basename(file_path)}")
            else:
                messagebox.showerror("Błąd", "Nie udało się zapisać obrazu.")

class ImageWindow(tk.Toplevel):
    def __init__(self, master, path):
        super().__init__(master)
        self.title(path.split("/")[-1])
        self.manager = ImageManager()
        self.manager.load_image(path)

        self.img_label = tk.Label(self)
        self.img_label.pack()

        self.bind("<FocusIn>", self.on_focus)

        self.display_image()

    def on_focus(self, event):
        self.master.set_active_window(self)

    def display_image(self):
        img = self.manager.get_display_image()
        if img is not None:
            self.tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.tk_img)
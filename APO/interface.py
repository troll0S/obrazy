import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from image_manager import ImageManager
from tkinter import messagebox
import os
import matplotlib.pyplot as plt


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

        histogram_menu = tk.Menu(menubar, tearoff=0)
        histogram_menu.add_command(label="show", command=self.show_histogram)
        histogram_menu.add_command(label="show LUT table", command=self.show_lut_table)
        histogram_menu.add_command(label="normalize", command=self.normalize)
        histogram_menu.add_command(label="equalize", command=self.equalize)

        image_menu = tk.Menu(menubar, tearoff=0)
        image_menu.add_command(label="RGB 2 Gray", command=self.rgb_to_gray)
        image_menu.add_command(label="RGB 2 3x Gray", command=self.rgb_to_3x_gray)
        image_menu.add_command(label="RGB 2 HSV", command=self.rgb_to_HSV)
        image_menu.add_command(label="RGB 2 Lab", command=self.rgb_to_lab)

        one_point_menu = tk.Menu(menubar,tearoff=0)
        one_point_menu.add_command(label="negate",command=self.negate)


        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Histogram", menu=histogram_menu)
        menubar.add_cascade(label="Image", menu=image_menu)
        menubar.add_cascade(label="One Point Operations",menu=one_point_menu)

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
        img = self.active_window.image_manager.get_display_image()
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

    def rgb_to_gray(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return

        self.active_window.manager.rgb_to_gray()
        self.active_window.display_image()

    def rgb_to_3x_gray(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        gray_images = self.active_window.manager.split_rgb_to_grays()
        if gray_images is None:
            messagebox.showwarning("Błąd", "Niepoprawny obraz.")
            return

        channel_names = ["Blue", "Green", "Red"]

        for i, img in enumerate(gray_images):
            window = tk.Toplevel(self)
            window.title(f"{channel_names[i]} channel as gray")

            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(display_img)
            tk_img = ImageTk.PhotoImage(pil_img)

            label = tk.Label(window, image=tk_img)
            label.image = tk_img
            label.pack()

    def rgb_to_HSV(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        if not self.active_window.manager.is_rgb():
            messagebox.showwarning("Błąd konwersji", "Obraz nie jest w formacie RGB. Konwersja do HSV niemożliwa.")
            return
        self.active_window.manager.rgb_to_HSV()
        self.active_window.display_image()

    def rgb_to_lab(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        if not self.active_window.manager.is_rgb():
            messagebox.showwarning("Błąd konwersji", "Obraz nie jest w formacie RGB. Konwersja do LAB niemożliwa.")
            return
        self.active_window.manager.rgb_to_lab()
        self.active_window.display_image()

    def show_histogram(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        if not self.active_window.manager.is_grayscale():
            messagebox.showinfo("Błąd", "Histogram można wyświetlić tylko dla obrazów w skali szarości.")
            return
        hist, bins = self.active_window.manager.get_histogram_data()
        if hist is None or bins is None:
            messagebox.showerror("Błąd", "Nie udało się wyliczyć histogramu.")
            return

        fig = plt.figure(f"Histogram_{id(self.active_window)}")
        max_val = hist.max()

        plt.xlabel("Wartość piksela")
        plt.ylim(0, max_val)
        plt.bar(bins[:-1], hist, width=1, color='gray')
        plt.grid(True)

        def on_close(event):
            self.active_window.manager.hist_window = None
            self.active_window.manager.hist_fig = None
            self.active_window.manager.histogram_shown = False

        self.active_window.manager.hist_window = fig
        self.active_window.manager.hist_fig = fig
        self.active_window.manager.histogram_shown = True

        fig.canvas.mpl_connect('close_event', on_close)
        plt.show(block=False)

    def show_lut_table(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        if not self.active_window.manager.is_grayscale():
            messagebox.showinfo("Błąd", "Histogram można wyświetlić tylko dla obrazów w skali szarości.")
            return

        result = self.active_window.manager.show_lut_table()
        if result is None:
            result = self.active_window.manager.get_lut()
            if result is None:
                messagebox.showerror("Błąd", "Nie udało się pobrać danych LUT.")
                return

        values,hist = result

        lut_window = tk.Toplevel(self)
        lut_window.title("LUT Table")
        self.active_window.manager.lut_window = lut_window

        canvas = tk.Canvas(lut_window)
        scrollbar = tk.Scrollbar(lut_window, orient="horizontal", command=canvas.xview)
        frame = tk.Frame(canvas)

        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(xscrollcommand=scrollbar.set)


        tk.Label(frame, text="Wartość", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5)
        tk.Label(frame, text="Liczba", font=("Arial", 10, "bold")).grid(row=1, column=0, padx=5)

        for i in range(256):
            tk.Label(frame, text=str(values[i])).grid(row=0, column=i + 1, padx=1)
            tk.Label(frame, text=str(hist[i])).grid(row=1, column=i + 1, padx=1)

        canvas.pack(fill="both", expand=True)
        scrollbar.pack(fill="x")

    def normalize(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        if not self.active_window.manager.is_grayscale():
            messagebox.showwarning("Błąd", "Normalizacja dostępna tylko dla obrazów w skali szarości.")
            return
        self.active_window.manager.normalize()
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def equalize(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        if not self.active_window.manager.is_grayscale():
            messagebox.showwarning("Błąd", "Normalizacja dostępna tylko dla obrazów w skali szarości.")
            return
        self.active_window.manager.equalize()
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def negate(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        if not self.active_window.manager.is_grayscale():
            messagebox.showwarning("Błąd", "Normalizacja dostępna tylko dla obrazów w skali szarości.")
            return
        self.active_window.manager.negate()
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def update_lut_and_histogram(self):
        if self.active_window is None:
            return
        if self.active_window.manager.is_histogram_active():
            self.show_histogram()
        if self.active_window.manager.is_lut_active():
            self.show_lut_table()

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
        self.zoom_level = 1.0
        self.create_window_menu()

    def on_focus(self, event):
        self.master.set_active_window(self)

    def display_image(self):
        img = self.manager.get_display_image()
        if img is not None:
            self.tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.tk_img)

    def create_window_menu(self):
        menu = tk.Menu(self)
        view_menu = tk.Menu(menu, tearoff=0)

        view_menu.add_command(label="Zoom In", command=self.zoom_in)
        view_menu.add_command(label="Zoom Out", command=self.zoom_out)
        view_menu.add_separator()
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom)

        menu.add_cascade(label="View", menu=view_menu)
        self.config(menu=menu)

    def zoom_in(self):
        self.zoom_level *= 2
        self.apply_zoom()

    def zoom_out(self):
        self.zoom_level *= 0.5
        self.apply_zoom()

    def reset_zoom(self):
        self.zoom_level = 1
        self.apply_zoom()

    def apply_zoom(self):
        self.manager.resize_current(self.zoom_level)
        self.display_image()

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import cv2
from image_manager import ImageManager
from tkinter import messagebox
import os
import matplotlib.pyplot as plt



class Interface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplikacja do przetwarzania obrazów")
        self.geometry("500x150")

        self.create_menubar()
        self.create_border_handling_options()
        self.windows = []
        self.active_window = None
        self.border_mode_map = {
            "constant": cv2.BORDER_CONSTANT,
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT,
            "reflect_101": cv2.BORDER_REFLECT_101,
            "wrap": cv2.BORDER_WRAP,
            "isolated": cv2.BORDER_ISOLATED
        }
        self.rect_start = None
        self.rect_end = None


    def create_menubar(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_command(label="Save Active Image", command=self.save_active_image)
        file_menu.add_command(label="Save with RLE")
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
        image_menu.add_command(label="Analizuj obiekt",command=self.analize_image)

        one_point_menu = tk.Menu(menubar,tearoff=0)
        one_point_menu.add_command(label="negate",command=self.negate)
        one_point_menu.add_command(label="Posterize", command=self.posterize)

        neighborhood_operations_menu = tk.Menu(menubar,tearoff=0)
        neighborhood_operations_menu.add_command(label="blur",command=self.apply_blur)
        neighborhood_operations_menu.add_command(label="gaussian blur",command=self.apply_goussian_blur)
        neighborhood_operations_menu.add_command(label="Sobel", command=self.apply_sobel)
        neighborhood_operations_menu.add_command(label="Laplacian",command=self.apply_laplacian)
        neighborhood_operations_menu.add_command(label="Canny",command=self.apply_canny)
        neighborhood_operations_menu.add_command(label="sharpen laplace cross",command=self.apply_sharpen_laplace_cross)
        neighborhood_operations_menu.add_command(label="sharpen laplace full",command=self.apply_sharpen_laplace_full)
        neighborhood_operations_menu.add_command(label="sharpen laplace extreme",command=self.apply_sharpen_laplace_extreme)

        prewitt_menu = tk.Menu(neighborhood_operations_menu,tearoff=0)
        prewitt_menu.add_command(label="n",command=self.apply_prewitt_n)
        prewitt_menu.add_command(label="nw",command=self.apply_prewitt_nw)
        prewitt_menu.add_command(label="w",command=self.apply_prewitt_w)
        prewitt_menu.add_command(label="sw",command=self.apply_prewitt_sw)
        prewitt_menu.add_command(label="s",command=self.apply_prewitt_s)
        prewitt_menu.add_command(label="se",command=self.apply_prewitt_se)
        prewitt_menu.add_command(label="e",command=self.apply_prewitt_e)
        prewitt_menu.add_command(label="ne",command=self.apply_prewitt_ne)

        neighborhood_operations_menu.add_cascade(label="Prewitt", menu=prewitt_menu)
        neighborhood_operations_menu.add_command(label="interactive mask", command=self.open_interactive_mask)

        median_menu = tk.Menu(neighborhood_operations_menu,tearoff=0)
        median_menu.add_command(label="3x3",command=self.apply_median_3x3)
        median_menu.add_command(label="5x5",command=self.apply_median_5x5)
        median_menu.add_command(label="7x7",command=self.apply_median_7x7)
        neighborhood_operations_menu.add_cascade(label="median", menu=median_menu)

        dual_operations_menu = tk.Menu(menubar, tearoff=0)
        dual_operations_menu.add_command(label="Add", command=lambda: self.dual_image_operation("add"))
        dual_operations_menu.add_command(label="Subtract", command=lambda: self.dual_image_operation("subtract"))
        dual_operations_menu.add_command(label="Blend", command=lambda: self.dual_image_operation("blend"))
        dual_operations_menu.add_command(label="Bitwise AND", command=lambda: self.dual_image_operation("and"))
        dual_operations_menu.add_command(label="Bitwise OR", command=lambda: self.dual_image_operation("or"))
        dual_operations_menu.add_command(label="Bitwise XOR", command=lambda: self.dual_image_operation("xor"))

        morphology_menu = tk.Menu(menubar, tearoff=0)
        morphology_menu.add_command(label="Open Morphology Settings", command=self.open_morphology_window)
        morphology_menu.add_command(label="Szkieletyzacja", command=self.skeletonize)
        morphology_menu.add_command(label="Hough transformation", command=self.hough_transformation)

        segmentation_menu = tk.Menu(menubar, tearoff=0)
        segmentation_menu.add_command(label="Manual threshhold", command=self.open_manual_threshold_window)
        segmentation_menu.add_command(label="Adaptive Threshold Mean", command=lambda: self.threshold_adaptive("mean"))
        segmentation_menu.add_command(label="Adaptive Threshold Gaussian", command=lambda: self.threshold_adaptive("gaussian"))
        segmentation_menu.add_command(label="Otsu Threshold", command=self.threshold_otsu)
        segmentation_menu.add_command(label="Grab cut", command=self.activate_grabcut)
        segmentation_menu.add_command(label="Watershed", command=self.watershed)
        segmentation_menu.add_command(label="inpainting", command=self.inpainting)


        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Histogram", menu=histogram_menu)
        menubar.add_cascade(label="Image", menu=image_menu)
        menubar.add_cascade(label="One Point Operations",menu=one_point_menu)
        menubar.add_cascade(label="Neighborhood Operations", menu=neighborhood_operations_menu)
        menubar.add_cascade(label="Dual Image Operations", menu=dual_operations_menu)
        menubar.add_cascade(label="LAB3", menu=morphology_menu)
        menubar.add_cascade(label="Segmentation", menu=segmentation_menu)

        self.config(menu=menubar)

    def create_border_handling_options(self):
        border_frame = tk.Frame(self)
        border_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        tk.Label(border_frame, text="Obsługa brzegów:").pack(side=tk.LEFT, padx=10)

        self.border_mode = tk.StringVar(value="reflect")

        options = ["constant","isolated", "reflect", "replicate","reflect_101","wrap"]
        border_menu = tk.OptionMenu(border_frame, self.border_mode, *options)
        border_menu.pack(side=tk.LEFT)

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

    def posterize(self,levels = 4):
        from tkinter import simpledialog
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        if not self.active_window.manager.is_grayscale():
            messagebox.showwarning("Błąd", "Normalizacja dostępna tylko dla obrazów w skali szarości.")
            return

        levels = simpledialog.askinteger("Posterize", "Podaj liczbę poziomów (2–256):", minvalue=2, maxvalue=256, initialvalue=4)
        if levels is None:
            return

        self.active_window.manager.posterize(levels)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_blur(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_blur(border_mode=mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_goussian_blur(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_gaussian_blur(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()


    def apply_sobel(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_sobel(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()


    def apply_laplacian(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_laplacian(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_canny(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_canny(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()


    def apply_sharpen_laplace_cross(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_sharpen_laplace_cross(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_sharpen_laplace_full(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_sharpen_laplace_full(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_sharpen_laplace_extreme(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_sharpen_laplace_extreme(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_prewitt_n(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_prewitt_n(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_prewitt_nw(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_prewitt_nw(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_prewitt_w(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_prewitt_w(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_prewitt_sw(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_prewitt_sw(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_prewitt_s(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_prewitt_s(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_prewitt_se(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_prewitt_se(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_prewitt_e(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_prewitt_e(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_prewitt_ne(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_prewitt_ne(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_median_3x3(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_median_3x3(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_median_5x5(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_median_5x5(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def apply_median_7x7(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_median_7x7(border_mode)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def open_interactive_mask(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        KernelDialog(self, self.on_mask_submitted)

    def on_mask_submitted(self, kernel):
        border_mode = self.border_mode_map[self.border_mode.get()]
        self.active_window.manager.apply_uniwersal(border_mode, kernel)
        self.active_window.display_image()
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def dual_image_operation(self,operation):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Nie wybrano aktywnego okna.")
            return
        path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.png *.bmp *.jpeg")])
        if not path:
            return

        alpha = 0.5
        if operation == "blend":
            alpha_str = tk.simpledialog.askstring("Mieszanie", "Podaj współczynnik alpha (0.0 - 1.0):", initialvalue="0.5")
            try:
               alpha = float(alpha_str)
            except (ValueError, TypeError):
                messagebox.showerror("Błąd", "Nieprawidłowa wartość alpha.")
                return
        self.active_window.manager.apply_dual_image_operation(path, operation, alpha)
        self.active_window.display_image()
        self.update_lut_and_histogram()

    def update_lut_and_histogram(self):
        if self.active_window is None:
            return
        if self.active_window.manager.is_histogram_active():
            self.show_histogram()
        if self.active_window.manager.is_lut_active():
            self.show_lut_table()

    def open_morphology_window(self):
        window = tk.Toplevel(self)
        window.title("Morphology Settings")
        window.geometry("350x300")

        # Lista opcji
        operations = ["Erozja", "Dylacja", "Otwarcie", "Zamknięcie"]
        shapes = ["Kwadrat", "Romb"]

        kernel_sizes = ["3x3", "5x5", "7x7"]

        # Zmienne
        op_var = tk.StringVar(value=operations[0])
        shape_var = tk.StringVar(value=shapes[0])
        size_var = tk.StringVar(value=kernel_sizes[0])

        # Interfejs
        tk.Label(window, text="Rodzaj operacji:").pack(pady=5)
        tk.OptionMenu(window, op_var, *operations).pack()

        tk.Label(window, text="Element strukturalny:").pack(pady=5)
        tk.OptionMenu(window, shape_var, *shapes).pack()


        tk.Label(window, text="Rozmiar jądra:").pack(pady=5)
        tk.OptionMenu(window, size_var, *kernel_sizes).pack()

        tk.Button(window, text="Apply", command=lambda: self.apply_morphology(
            op_var.get(), shape_var.get(), self.border_mode.get(), size_var.get())).pack(pady=10)

    def apply_morphology(self, operation, shape, border, kernel_size):
        if self.active_window is None:
            messagebox.showwarning("Brak aktywnego obrazu", "Najpierw wczytaj obraz.")
            return
        border_str = border
        border_code = self.border_mode_map.get(border_str, cv2.BORDER_REFLECT)
        try:
            k = int(kernel_size[0])  # np. "3x3" → 3

            self.active_window.manager.apply_morphology(
                operation=operation,
                shape=shape,
                border=border_code,
                kernel_size=k
            )
            self.active_window.display_image()
            self.update_lut_and_histogram()

        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił problem podczas przetwarzania:\n{e}")

    def skeletonize(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Najpierw załaduj obraz.")
            return

        try:
            if not self.active_window.manager.is_binary():
                messagebox.showwarning("Obraz niebinarny", "Szkieletyzacja wymaga obrazu binarnego (tylko 0 i 255).")
                return
            border_str = self.border_mode.get()
            border_code = self.border_mode_map.get(border_str, cv2.BORDER_REFLECT)
            self.active_window.manager.skeletonize(border_code)
            self.update_lut_and_histogram()
            self.active_window.display_image()

        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd podczas szkieletyzacji:\n{e}")

    def hough_transformation(self):
        if self.active_window is None:
            messagebox.showinfo("Brak aktywnego obrazu", "Najpierw załaduj obraz.")
            return
        self.active_window.manager.hough_transformation()
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def threshold_adaptive(self,method):
        if self.active_window is None:
            messagebox.showerror("Błąd", "Brak aktywnego obrazu.")
            return
        if method == "mean":
            self.active_window.manager.threshold_adaptive(cv2.ADAPTIVE_THRESH_MEAN_C)
        elif method == "gaussian":
            self.active_window.manager.threshold_adaptive(cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        else:
            messagebox.showerror("Błąd", "niepoprawna metoda threshhold")
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def threshold_otsu(self):
        if self.active_window is None:
            messagebox.showerror("Błąd", "Brak aktywnego obrazu.")
        self.active_window.manager.threshold_otsu()
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def open_manual_threshold_window(self):
        if self.active_window is None:
            messagebox.showerror("Błąd", "Brak aktywnego obrazu.")
            return
        original_image = self.active_window.manager.current.copy()

        window = tk.Toplevel()
        window.title("Manual Threshold")
        reverse_var = tk.BooleanVar()
        reverse_check = tk.Checkbutton(window, text="Reverse", variable=reverse_var)
        reverse_check.pack(pady=5)
        slider = tk.Scale(window, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold value")
        slider.set(127)
        slider.pack(padx=10, pady=5, fill='x')

        def update_image(thresh_value):
            try:
                thresh_value = int(thresh_value)
                self.active_window.manager.set_current(original_image)
                if reverse_var.get():
                    self.active_window.manager.threshold_manual_reverse(thresh_value)
                else:
                    self.active_window.manager.threshold_manual(thresh_value)

            except Exception as e:
                print("Błąd przy progowaniu:", e)
            self.update_lut_and_histogram()
            self.active_window.display_image()

        slider.configure(command=update_image)

        def on_reverse_toggle(*args):
            update_image(slider.get())

        reverse_var.trace_add("write", on_reverse_toggle)
        button_frame = tk.Frame(window)
        button_frame.pack(pady=10)

        def close_and_restore():
            self.active_window.manager.set_current(original_image)
            window.destroy()
            self.update_lut_and_histogram()
            self.active_window.display_image()

        def apply_and_close():
            window.destroy()
            self.update_lut_and_histogram()
            self.active_window.display_image()

        close_button = tk.Button(button_frame, text="Zamknij", command=close_and_restore)
        close_button.pack(side=tk.LEFT, padx=5)

        apply_button = tk.Button(button_frame, text="Zastosuj", command=apply_and_close)
        apply_button.pack(side=tk.LEFT, padx=5)
        update_image(127)

    def activate_grabcut(self):
        if self.active_window is None:
            messagebox.showwarning("Brak obrazu", "Najpierw załaduj obraz.")
            return

        label = self.active_window.img_label
        label.bind("<ButtonPress-1>", self.start_rectangle)
        label.bind("<B1-Motion>", self.draw_rectangle)
        label.bind("<ButtonRelease-1>", self.finish_rectangle)
        messagebox.showinfo("GrabCut", "Zaznacz prostokąt na obrazie, trzymając lewy przycisk myszy.")
    def start_rectangle(self, event):
        self.rect_start = (event.x, event.y)
        self.rect_end = None

    def draw_rectangle(self, event):
        if self.rect_start:
            x0, y0 = self.rect_start
            x1, y1 = event.x, event.y
            self.draw_temp_rectangle(x0, y0, x1, y1)

    def finish_rectangle(self, event):
        if not self.rect_start:
            return

        self.rect_end = (event.x, event.y)
        label = self.active_window.img_label
        label.unbind("<ButtonPress-1>")
        label.unbind("<B1-Motion>")
        label.unbind("<ButtonRelease-1>")
        if messagebox.askyesno("GrabCut", "Zastosować GrabCut na zaznaczonym obszarze?"):
            self.apply_grabcut()
        else:
            self.active_window.display_image()  # Przywróć oryginalny obraz
            self.rect_start = None
            self.rect_end = None


    def apply_grabcut(self):
        if not self.rect_start or not self.rect_end:
            messagebox.showerror("Błąd", "Niepoprawne zaznaczenie prostokąta.")
            return

        x0, y0 = self.rect_start
        x1, y1 = self.rect_end
        rect = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

        self.active_window.manager.grabcut(rect)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def draw_temp_rectangle(self, x0, y0, x1, y1):
        # Pobierz aktualny obraz jako kopię
        base_img = self.active_window.manager.get_display_image().copy()

        # Narysuj prostokąt na obrazie
        draw = ImageDraw.Draw(base_img)
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

        # Zaktualizuj label w ImageWindow
        self.active_window.tk_img = ImageTk.PhotoImage(base_img)
        self.active_window.img_label.config(image=self.active_window.tk_img)

    def watershed(self):
        if self.active_window is None:
            messagebox.showwarning("Brak obrazu", "Najpierw załaduj obraz.")
            return

        self.active_window.manager.watershed()
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def inpainting(self):
        if self.active_window is None:
            messagebox.showwarning("Brak obrazu", "Najpierw załaduj obraz.")
            return
        path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.png *.bmp *.jpeg")])
        if not path:
            return

        self.active_window.manager.inpainting(path)
        self.update_lut_and_histogram()
        self.active_window.display_image()

    def analize_image(self):
        if self.active_window is None:
            messagebox.showwarning("Brak obrazu", "Najpierw załaduj obraz.")
            return
        data = self.active_window.manager.analize_image()
        self.present_analize_data(data)

    def present_analize_data(self,data):
        window = tk.Toplevel(self)
        window.title("Wyniki analizy obiektów")
        headers = [
            "Obiekt", "Moment m00", "Centroid (cx, cy)", "Pole", "Obwód",
            "Aspect Ratio", "Extent", "Solidity", "Equivalent Diameter"
        ]
        for col, header in enumerate(headers):
            tk.Label(window, text=header, font=("Arial", 10, "bold")).grid(row=0, column=col, padx=5, pady=5)
        for row_idx, obj in enumerate(data, start=1):
            tk.Label(window, text=str(row_idx)).grid(row=row_idx, column=0, padx=5, pady=2)
            tk.Label(window, text=f"{obj['m00']:.2f}").grid(row=row_idx, column=1)
            tk.Label(window, text=f"({obj['cx']:.1f}, {obj['cy']:.1f})").grid(row=row_idx, column=2)
            tk.Label(window, text=f"{obj['area']:.2f}").grid(row=row_idx, column=3)
            tk.Label(window, text=f"{obj['perimeter']:.2f}").grid(row=row_idx, column=4)
            tk.Label(window, text=f"{obj['aspect_ratio']:.2f}").grid(row=row_idx, column=5)
            tk.Label(window, text=f"{obj['extent']:.2f}").grid(row=row_idx, column=6)
            tk.Label(window, text=f"{obj['solidity']:.2f}").grid(row=row_idx, column=7)
            tk.Label(window, text=f"{obj['equivalent_diameter']:.2f}").grid(row=row_idx, column=8)

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
        self.pyramid_level = 0



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
        view_menu.add_command(label="reset image", command=self.reset_image)

        pyr_menu=tk.Menu(menu, tearoff=0)

        pyr_menu.add_command(label="pyramid Up", command=self.pyramid_Up)
        pyr_menu.add_command(label="pyramid Down", command=self.pyramid_Down)
        pyr_menu.add_command(label="pyramid Reset", command=self.pyramid_Reset)

        menu.add_cascade(label="View", menu=view_menu)
        menu.add_cascade(label="Pyramid", menu=pyr_menu)
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

    def reset_image(self):
        self.manager.reset_image()
        self.display_image()

    def pyramid_Up(self):
        self.pyramid_level += 1
        if self.pyramid_level <= 2:
            self.manager.pyramid_Up()
            self.display_image()
        else:
            self.pyramid_level = 2
            messagebox.showinfo("limit piramidy", "osiągnięto limit piramidy (max 2 w góre i 2 w dół).")

    def pyramid_Down(self):
        self.pyramid_level -= 1
        if self.pyramid_level >= -2:
            self.manager.pyramid_Down()
            self.display_image()
        else:
            self.pyramid_level = -2
            messagebox.showinfo("limit piramidy", "osiągnięto limit piramidy (max 2 w góre i 2 w dół).")

    def pyramid_Reset(self):
        self.pyramid_level = 0
        self.manager.pyramid_Reset()
        self.display_image()


class KernelDialog(tk.Toplevel):
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.title("Wprowadź maskę 3x3")
        self.entries = []
        self.callback = callback

        for i in range(3):
            row = []
            for j in range(3):
                entry = tk.Entry(self, width=5, justify='center')
                entry.grid(row=i, column=j, padx=5, pady=5)
                entry.insert(0, "0")
                row.append(entry)
            self.entries.append(row)

        submit_btn = tk.Button(self, text="Zastosuj", command=self.on_submit)
        submit_btn.grid(row=3, column=0, columnspan=3, pady=10)

    def on_submit(self):
        try:
            kernel = [
                [float(self.entries[i][j].get()) for j in range(3)]
                for i in range(3)
            ]
            self.callback(kernel)
            self.destroy()
        except ValueError:
            tk.messagebox.showerror("Błąd", "Wszystkie pola muszą zawierać liczby.")
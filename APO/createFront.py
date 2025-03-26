import tkinter as tk
from tkinter import filedialog, ttk



class createFront:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.geometry("800x600")

        self.create_menu()
        self.create_main_frame()

        #self.image_label = tk.Label(self.main_frame, text="No Image Loaded", bg="gray", width=50, height=20)
        #self.image_label.pack()



    def create_menu(self):
        menu_bar = tk.Menu(self.root)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Load Image")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        processing_menu = tk.Menu(menu_bar, tearoff=0)
        processing_menu.add_command(label="Histogram")
        processing_menu.add_command(label="Stretch Histogram")
        processing_menu.add_command(label="Equalize Histogram")
        processing_menu.add_command(label="Selective Stretching")
        menu_bar.add_cascade(label="Histogram Processing", menu=processing_menu)

        filtering_menu = tk.Menu(menu_bar, tearoff=0)
        filtering_menu.add_command(label="Smoothing Mask")
        filtering_menu.add_command(label="Sharpening Mask")
        filtering_menu.add_command(label="Median Operations")
        menu_bar.add_cascade(label="Filtering", menu=filtering_menu)

        conversion_menu = tk.Menu(menu_bar, tearoff=0)
        conversion_menu.add_command(label="RGB to Grayscale Channels")
        conversion_menu.add_command(label="Posterization")
        menu_bar.add_cascade(label="Conversion & Transformations", menu=conversion_menu)

        morphology_menu = tk.Menu(menu_bar, tearoff=0)
        morphology_menu.add_command(label="Skeletonization")
        morphology_menu.add_command(label="Profile Line")
        morphology_menu.add_command(label="Hough Transform")
        menu_bar.add_cascade(label="Morphology", menu=morphology_menu)

        multi_scale_menu = tk.Menu(menu_bar, tearoff=0)
        multi_scale_menu.add_command(label="Image Pyramids")
        menu_bar.add_cascade(label="Multi-Scale Processing", menu=multi_scale_menu)

        segmentation_menu = tk.Menu(menu_bar, tearoff=0)
        segmentation_menu.add_command(label="Segmentation")
        segmentation_menu.add_command(label="GrabCut")
        segmentation_menu.add_command(label="Watershed")
        menu_bar.add_cascade(label="Segmentation", menu=segmentation_menu)

        feature_menu = tk.Menu(menu_bar, tearoff=0)
        feature_menu.add_command(label="Feature Vector")
        feature_menu.add_command(label="RLE Compression")
        menu_bar.add_cascade(label="Feature Extraction & Compression", menu=feature_menu)

        reconstruction_menu = tk.Menu(menu_bar, tearoff=0)
        reconstruction_menu.add_command(label="Image Inpainting")
        menu_bar.add_cascade(label="Image Reconstruction", menu=reconstruction_menu)

        self.root.config(menu=menu_bar)

    def create_main_frame(self):
        self.main_frame = tk.Frame(self.root, bg="white")
        self.main_frame.pack(fill=tk.BOTH, expand=True)




if __name__ == "__main__":
    root = tk.Tk()
    root.mainloop()

# APO – Image Processing App

A desktop application for image processing and analysis, built in Python with OpenCV and a Tkinter GUI. Created as a university project for the *Image Processing and Analysis* (APO) course.

<!-- Add a screenshot here, e.g. ![Screenshot](docs/screenshot.png) -->

## Features

- **Histogram and LUT:** histogram view, LUT table, normalisation (stretching), equalisation
- **Colour spaces:** RGB → grayscale, RGB → 3 grayscale channels, RGB → HSV, RGB → Lab
- **Point operations:** negation, posterisation, manual, inverted, Otsu and adaptive thresholding
- **Neighbourhood operations:** blur, Gaussian blur, median 3×3 / 5×5 / 7×7, Laplacian sharpening, edge detection (Sobel, Laplacian, Canny, Prewitt in 8 directions), custom kernel with selectable border handling
- **Two-image operations:** add, subtract, blend, bitwise AND / OR / XOR
- **Morphology:** erosion, dilation, opening, closing with configurable kernel shape and size; skeletonisation
- **Segmentation and analysis:** Hough line transform, GrabCut, watershed, inpainting, object analysis (area, perimeter, centroid, aspect ratio, extent, solidity, equivalent diameter)
- **Other:** image pyramid (up / down), zoom, RLE compression ratio

## Tech stack

Python 3.12 · OpenCV · NumPy · Pillow · Matplotlib · Tkinter · PyInstaller (Windows build)

## Getting started

```bash
git clone https://github.com/troll0S/obrazy.git
cd obrazy/APO
pip install -r requirements.txt
python main.py
```

## Project structure

```
APO/
├── main.py           # entry point
├── interface.py      # Tkinter GUI: menus, dialogs, image display
└── image_manager.py  # image processing logic (OpenCV / NumPy)
```

The GUI layer (`interface.py`) is separated from the processing logic (`image_manager.py`), so operations can be tested or reused without the interface.

## Author

Wojciech Jurek – [GitHub](https://github.com/troll0S) · [LinkedIn](https://www.linkedin.com/in/wojciech-jurek-825027339)

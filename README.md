# Automated Logistics OCR

**Automated Logistics OCR** is an intelligent CV pipeline designed to extract and digitize text from packaging labels. It handles real-world challenges like low lighting, image noise, and rotation to provide reliable Optical Character Recognition (OCR) for logistics contexts.

## 🚀 Features

-   **Robost Preprocessing**: Uses bilateral filtering and adaptive thresholding to clean grainy or noisy images.
-   **Smart Label Detection**: Automatically detects rectangular label regions on boxes to focus OCR only where it matters.
-   **Synthetic Data Engine**: Includes a generator (`generate_data.py`) to create synthetic training data with realistic noise, rotation, and artifacts for benchmarking.
-   **EasyOCR Integration**: Leveraging deep learning-based OCR for high accuracy.

## 🛠️ Installation

Ensure you have Python installed. Clone the repository and install the dependencies:

```bash
pip install opencv-python numpy easyocr pandas
```

*Note: For GPU acceleration with EasyOCR, ensure you have the appropriate CUDA drivers installed and configured.*

## 📂 Project Structure

```
box-text-reader/
├── data/
│   ├── synthetic/       # Generated synthetic images
│   └── labels.csv       # Ground truth for synthetic data
├── pipeline.py          # Main OCR pipeline logic
├── generate_data.py     # Script to generate synthetic box images
└── test_real_image.py   # Script to test on single real images
```

## 💻 Usage

### 1. Generate Synthetic Data
To test the pipeline without real photos, generate a batch of synthetic box images:

```bash
python generate_data.py
```
This will create a `data/synthetic` folder with 10 sample images and a `labels.csv` file.

### 2. Run the Pipeline
Process all images in the `data/synthetic` directory (or point it to your own data directory in the code):

```bash
python pipeline.py
```
The script will:
1.  Read images from the input directory.
2.  Preprocess and detect labels.
3.  Extract text.
4.  Save the results to `ocr_results.csv`.

## 🧠 How It Works

1.  **Preprocessing**: content is converted to grayscale, denoised using a bilateral filter (preserving edges), and then binary thresholded using Adaptive Gaussian Thresholding.
2.  **Label Localization**: The system finds contours in the thresholded image, filtering for quadrilateral shapes that match the aspect ratio and area constraints of a typical shipping label.
3.  **Extraction**: The detected region is cropped and passed to EasyOCR.
4.  **Fallback**: If no specific label region is found, the system attempts to read the entire image as a fallback.

## 🔮 Future Improvements

-   Add support for reading barcodes/QR codes alongside text.
-   Implement a deep learning-based object detector (YOLO) for more robust label localization in cluttered scenes.
-   expose as a FastAPI endpoint.

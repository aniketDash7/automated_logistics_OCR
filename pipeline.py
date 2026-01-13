import cv2
import numpy as np
import easyocr
import pandas as pd
import os
import glob

class BoxTextReader:
    def __init__(self):
        # Initialize EasyOCR reader (only need to do this once)
        print("Initializing OCR Reader...")
        self.reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have a GPU
        
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Noise reduction (Bilateral filter is good for keeping edges sharp)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive Thresholding to handle lighting/noise
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return img, thresh

    def find_label_region(self, thresh):
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        label_contours = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            
            # Label is usually a rectangle (4 points)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h
                img_area = thresh.shape[0] * thresh.shape[1]
                cnt_area = cv2.contourArea(cnt)
                
                # Filter by relative area (0.1% to 10% of image) and aspect ratio
                if (0.001 * img_area < cnt_area < 0.2 * img_area) and (0.5 < aspect_ratio < 4.0):
                    label_contours.append(cnt)
        
        # Return the contour with the smallest y (assuming it's nested or we want the one highest up)
        # or simply the one with largest area if multiple found
        if not label_contours:
            return None
            
        return max(label_contours, key=cv2.contourArea)

    def process_image(self, image_path):
        img, thresh = self.preprocess_image(image_path)
        if img is None:
            return None
            
        label_cnt = self.find_label_region(thresh)
        
        if label_cnt is not None:
            x, y, w, h = cv2.boundingRect(label_cnt)
            # Add small padding
            pad = 5
            crop = img[max(0, y-pad):min(img.shape[0], y+h+pad), 
                       max(0, x-pad):min(img.shape[1], x+w+pad)]
            
            # OCR
            results = self.reader.readtext(crop)
            if results:
                # Combine all text found in the label area
                text = " ".join([res[1] for res in results])
                return text.strip()
        
        # Fallback: OCR on entire image if label detection fails (slower but safer)
        results = self.reader.readtext(img)
        if results:
            # Look for strings that match our "box" pattern? 
            # For now, just return first result
            return results[0][1]
            
        return "Not Found"

def main():
    pipeline = BoxTextReader()
    input_dir = "data/synthetic"
    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    
    results_list = []
    
    print(f"Processing {len(image_paths)} images...")
    for path in image_paths:
        filename = os.path.basename(path)
        print(f"Processing {filename}...")
        extracted_text = pipeline.process_image(path)
        results_list.append({"filename": filename, "extracted_text": extracted_text})
        
    # Save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv("ocr_results.csv", index=False)
    print("Results saved to ocr_results.csv")

if __name__ == "__main__":
    main()

import cv2
import os
from pipeline import BoxTextReader

def test_single_image(image_path):
    print(f"Testing on image: {image_path}")
    
    if not os.path.exists(image_path):
        print("Error: Image file not found.")
        return

    reader = BoxTextReader()
    
    # Process
    try:
        text = reader.process_image(image_path)
        print(f"\n--- Result ---\nDetected Text: '{text}'\n--------------")
        
        # Visualize for debugging (optional, saves to verify what it saw)
        img, thresh = reader.preprocess_image(image_path)
        label_cnt = reader.find_label_region(thresh)
        
        if label_cnt is not None:
            # Draw contour on original image
            debug_img = img.copy()
            x, y, w, h = cv2.boundingRect(label_cnt)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(debug_img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imwrite("debug_real_image_result.jpg", debug_img)
            print("Saved debug visualization to debug_real_image_result.jpg")
        else:
            print("Warning: Label region not explicitly detected using contours. Falling back to full image OCR.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Path to the user's uploaded image
    img_path = r"C:/Users/anike/.gemini/antigravity/brain/03866dc3-acbb-4475-871c-b3fb05e4741a/uploaded_image_1768110482275.jpg"
    test_single_image(img_path)

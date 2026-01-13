import cv2
import numpy as np
import random
import string
import os
from PIL import Image, ImageDraw, ImageFont

def generate_random_text(length=8):
    letters_and_digits = string.ascii_uppercase + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

def add_noise(image):
    # Gaussian noise
    row, col = image.shape
    mean = 0
    var = random.uniform(10, 50)
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    
    # Salt and pepper noise
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(noisy)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in [row, col]]
    out[tuple(coords)] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in [row, col]]
    out[tuple(coords)] = 0
    
    return np.clip(out, 0, 255).astype(np.uint8)

def generate_synthetic_image(output_path, text):
    # Image dimensions
    width, height = 800, 800
    
    # Create background (white)
    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)
    
    # Draw a square (box)
    box_size = random.randint(400, 600)
    box_x = random.randint(50, width - box_size - 50)
    box_y = random.randint(50, height - box_size - 50)
    box_coords = [box_x, box_y, box_x + box_size, box_y + box_size]
    draw.rectangle(box_coords, outline=50, width=5)
    
    # Draw a rectangle (label) inside the box
    label_w = random.randint(150, 300)
    label_h = random.randint(60, 100)
    label_x = random.randint(box_x + 20, box_x + box_size - label_w - 20)
    label_y = random.randint(box_y + 20, box_y + box_size - label_h - 20)
    label_coords = [label_x, label_y, label_x + label_w, label_y + label_h]
    draw.rectangle(label_coords, outline=0, width=3)
    
    # Draw text on the label
    # Try to load a default font
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
        
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    text_x = label_x + (label_w - text_w) // 2
    text_y = label_y + (label_h - text_h) // 2
    draw.text((text_x, text_y), text, fill=0, font=font)
    
    # Convert to numpy and rotate randomly
    img_np = np.array(img)
    angle = random.randint(-15, 15)
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_np = cv2.warpAffine(img_np, M, (width, height), borderValue=255)
    
    # Add noise
    img_np = add_noise(img_np)
    
    # Save
    cv2.imwrite(output_path, img_np)
    return text

if __name__ == "__main__":
    os.makedirs("data/synthetic", exist_ok=True)
    with open("data/labels.csv", "w") as f:
        f.write("filename,text\n")
        for i in range(10): # Generate 10 samples for testing
            filename = f"box_{i}.png"
            text = generate_random_text()
            generate_synthetic_image(f"data/synthetic/{filename}", text)
            f.write(f"{filename},{text}\n")
    print("Generated 10 synthetic images in data/synthetic/")

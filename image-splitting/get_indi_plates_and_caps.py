import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

def split_image(image_path, rows=6, cols=5, output_dir='split_images'):
    # Read the image
    img = cv2.imread(image_path)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Calculate the size of each cell
    cell_height = height // rows
    cell_width = width // cols
    
    # Calculate padding to ensure circles are centered
    pad_y = cell_height // 4
    pad_x = cell_width // 4
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the image and save each cell
    splits = []
    for i in range(rows):
        for j in range(cols):
            # Calculate cell boundaries with padding
            start_y = max(0, i * cell_height - pad_y)
            end_y = min(height, (i + 1) * cell_height + pad_y)
            start_x = max(0, j * cell_width - pad_x)
            end_x = min(width, (j + 1) * cell_width + pad_x)
            
            cell = img[start_y:end_y, start_x:end_x]
            splits.append(cell)
            
            # Save the split image
            output_path = os.path.join(output_dir, f'cell_{i+1}_{j+1}.png')
            cv2.imwrite(output_path, cell)
    
    return splits, output_dir

def process_single_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Use pytesseract to do OCR
    text = pytesseract.image_to_string(Image.fromarray(thresh), config='--psm 6')
    
    # Extract number from text
    number = ''.join(filter(str.isdigit, text))
    
    # Convert number to vector
    if number:
        vector = [int(digit) for digit in number]
    else:
        vector = []
    
    return vector

if __name__ == "__main__":
    image_path = "image-splitting/C1_1-30.png"
    
    # Step 1: Split the image and save subimages
    split_images, output_dir = split_image(image_path)
    print(f"Image split into {len(split_images)} parts")
    print(f"Subimages saved in directory: {output_dir}")
    
    # Step 2: Process a single image (let's use the first split as an example)
    vector = process_single_image(split_images[0])
    print(f"Extracted vector: {vector}")


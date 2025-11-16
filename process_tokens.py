"""
Token Image Background Remover

This script processes circular token images, identifies the background,
and makes it transparent. It uses edge detection to distinguish the token
from the background.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path


def find_circular_mask(image):
    """
    Detect the circular token in the image and create a mask.
    
    Args:
        image: BGR image from cv2
        
    Returns:
        tuple: (mask, circle_info) where mask is binary and circle_info contains {x, y, radius}
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection to find strong edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to connect nearby contours
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask
    mask = np.zeros(gray.shape, dtype=np.uint8)
    circle_info = None
    
    if contours:
        # Find the largest contour (assumed to be the token)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get circle information before eroding
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        # Calculate circularity: 4π * area / perimeter^2
        # Perfect circle = 1.0, less circular shapes approach 0
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Fit ellipse to detect scaling/stretching
        ellipse_info = None
        if len(largest_contour) >= 5:  # Need at least 5 points to fit ellipse
            ellipse = cv2.fitEllipse(largest_contour)
            (ex, ey), (minor_axis, major_axis), angle = ellipse
            # Ensure major >= minor
            if minor_axis > major_axis:
                minor_axis, major_axis = major_axis, minor_axis
            
            # Calculate aspect ratio (1.0 = circle, >1.0 = ellipse)
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
            
            ellipse_info = {
                'major_axis': float(major_axis),
                'minor_axis': float(minor_axis),
                'aspect_ratio': float(aspect_ratio),
                'angle': float(angle)  # Angle of major axis from horizontal
            }
        
        circle_info = {
            'x': float(x),
            'y': float(y),
            'radius': float(radius),
            'circularity': float(circularity),
            'ellipse': ellipse_info
        }
        
        # Fill the contour on mask
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Erode the mask to remove background edge pixels
        # Increase iterations to remove more edge pixels
        erode_kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, erode_kernel, iterations=5)
    
    return mask, circle_info


def remove_background(image_path, output_path):
    """
    Process a single image: detect circular token and make background transparent.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        
    Returns:
        dict or None: Image metadata if successful, None if failed
    """
    # Read image
    img = cv2.imread(str(image_path))
    
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Find the circular mask and circle info
    mask, circle_info = find_circular_mask(img)
    
    # Apply morphological operations to smooth the mask edges
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Calculate average color of outermost ring (border pixels touching transparent area)
    # Find the outer edge by dilating the mask and subtracting original
    dilate_kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    outer_ring = cv2.subtract(dilated_mask, mask)
    
    # Get pixels in the outer ring
    outer_pixels = img[outer_ring > 0]
    
    # Calculate average color (BGR format in OpenCV)
    if len(outer_pixels) > 0:
        avg_color_bgr = outer_pixels.mean(axis=0)
        avg_color_rgb = {
            'r': int(avg_color_bgr[2]),  # Convert BGR to RGB
            'g': int(avg_color_bgr[1]),
            'b': int(avg_color_bgr[0]),
            'hex': f"#{int(avg_color_bgr[2]):02x}{int(avg_color_bgr[1]):02x}{int(avg_color_bgr[0]):02x}"
        }
    else:
        avg_color_rgb = {'r': 0, 'g': 0, 'b': 0, 'hex': '#000000'}
    
    # Convert image to RGBA
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # Apply mask to alpha channel
    img_rgba[:, :, 3] = mask
    
    # Save result
    cv2.imwrite(str(output_path), img_rgba)
    print(f"Processed: {image_path.name} -> {output_path.name}")
    
    # Create metadata
    ellipse = circle_info.get('ellipse') if circle_info else None
    metadata = {
        'filename': image_path.name,
        'xsize': width,
        'ysize': height,
        'xpos': round(circle_info['x'], 1) if circle_info else 0,
        'ypos': round(circle_info['y'], 1) if circle_info else 0,
        'diameter': round(circle_info['radius'] * 2, 1) if circle_info else 0,
        'circularity': round(circle_info['circularity'], 3) if circle_info else 0,
        'aspect_ratio': round(ellipse['aspect_ratio'], 3) if ellipse else 1.0,
        'major_axis': round(ellipse['major_axis'], 1) if ellipse else 0,
        'minor_axis': round(ellipse['minor_axis'], 1) if ellipse else 0,
        'ellipse_angle': round(ellipse['angle'], 1) if ellipse else 0,
        'border_color': avg_color_rgb
    }
    
    return metadata


def process_all_images(input_folder="img/input", output_subfolder="img/output"):
    """
    Process all images in the input folder and save to output folder.
    
    Args:
        input_folder: Folder containing input images
        output_subfolder: Subfolder name for output images
    """
    # Get paths
    script_dir = Path(__file__).parent
    input_path = script_dir / input_folder
    output_path = script_dir / output_subfolder
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Find all images in input folder
    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {input_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print("-" * 50)
    
    # Process each image and collect metadata
    image_metadata = []
    success_count = 0
    for image_file in image_files:
        # Create output filename with .png extension (to support transparency)
        output_file = output_path / f"{image_file.stem}.png"
        
        metadata = remove_background(image_file, output_file)
        if metadata:
            image_metadata.append(metadata)
            success_count += 1
    
    # Save metadata to JSON file in img folder (parent of input folder)
    if image_metadata:
        img_folder = input_path.parent
        json_path = img_folder / "image_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(image_metadata, f, indent=2)
        print("-" * 50)
        print(f"Metadata saved to: {json_path}")
        
        # Create print format template
        print_template = {
            "settings": {
                "print_width": 8.5,
                "print_height": 11.0,
                "x_margin": 0.5,
                "y_margin": 0.5,
                "x_spacer": 0.2,
                "y_spacer": 0.2,
                "token_size": 1.0,
                "ppi": 300
            },
            "tokens_quantity_list": [
                {"filename": meta["filename"], "quantity": 1}
                for meta in image_metadata
            ]
        }
        
        template_path = img_folder / "print_format_template.json"
        with open(template_path, 'w') as f:
            json.dump(print_template, f, indent=2)
        print(f"Print template saved to: {template_path}")
    
    print("-" * 50)
    print(f"Successfully processed {success_count} out of {len(image_files)} images")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    process_all_images()

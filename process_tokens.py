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


def calculate_radial_variance(contour, center):
    """
    Calculate radial variance to distinguish circles from rounded squares.
    Measures distance from center to contour edge at many angles.
    
    Args:
        contour: OpenCV contour points
        center: (x, y) tuple of shape center
        
    Returns:
        dict: {
            'radial_std': standard deviation of radii (normalized),
            'radial_variance_ratio': variance relative to mean radius,
            'is_circle': True if shape is circular (low radial variance)
        }
    """
    cx, cy = center
    num_angles = 360  # Sample every degree
    
    radii = []
    for angle_deg in range(num_angles):
        angle_rad = np.deg2rad(angle_deg)
        # Create a ray from center in this direction
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        # Find intersection with contour by sampling along ray
        # Use maximum image dimension as search distance
        max_dist = 2000
        best_dist = 0
        
        for dist in range(10, max_dist, 2):
            point = np.array([cx, cy]) + direction * dist
            px, py = int(point[0]), int(point[1])
            
            # Check if this point is on the contour
            result = cv2.pointPolygonTest(contour, (float(px), float(py)), False)
            if result >= 0:  # Inside or on the contour
                best_dist = dist
            elif best_dist > 0:  # Was inside, now outside - found edge
                break
        
        if best_dist > 0:
            radii.append(best_dist)
    
    if len(radii) < 180:  # Need at least half the samples
        return {
            'radial_std': 0,
            'radial_variance_ratio': 0,
            'is_circle': False
        }
    
    radii = np.array(radii)
    mean_radius = np.mean(radii)
    std_radius = np.std(radii)
    variance_ratio = std_radius / mean_radius if mean_radius > 0 else 0
    
    # Circles have low radial variance (< 3% of mean radius)
    # Rounded squares have higher variance due to flat sides
    is_circle = variance_ratio < 0.03
    
    return {
        'radial_std': float(std_radius),
        'radial_variance_ratio': float(variance_ratio),
        'is_circle': bool(is_circle)
    }


def find_circular_mask(image):
    """
    Detect the circular token in the image and create a mask.
    Uses color-based detection to find background, then refines with edge detection.
    
    Args:
        image: BGR image from cv2
        
    Returns:
        tuple: (mask, circle_info) where mask is binary and circle_info contains {x, y, radius}
    """
    height, width = image.shape[:2]
    
    # Sample corner pixels to determine background color
    # Average the four corners (assumed to be background)
    corner_size = 20
    corners = [
        image[0:corner_size, 0:corner_size],  # Top-left
        image[0:corner_size, width-corner_size:width],  # Top-right
        image[height-corner_size:height, 0:corner_size],  # Bottom-left
        image[height-corner_size:height, width-corner_size:width]  # Bottom-right
    ]
    
    corner_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
    bg_color = corner_pixels.mean(axis=0)
    
    # Calculate color difference from background
    color_diff = np.sqrt(np.sum((image.astype(float) - bg_color) ** 2, axis=2))
    
    # Threshold to create initial mask (anything significantly different from background)
    # Use adaptive threshold based on color variance
    threshold = max(30, color_diff.std() * 1.5)
    mask = (color_diff > threshold).astype(np.uint8) * 255
    
    # Clean up the mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours on the color-based mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create final mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
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
        
        # Calculate radial variance to distinguish circles from rounded squares
        # This is more robust than perimeter-based circularity for rough/textured borders
        radial_info = calculate_radial_variance(largest_contour, (x, y))
        
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
            'radial_variance': radial_info['radial_variance_ratio'],
            'is_radially_circular': radial_info['is_circle'],
            'ellipse': ellipse_info
        }
        
        # Fill the contour on mask
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Apply minimal smoothing to the mask edges
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
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
    
    # Calculate border color at 3px depth from edge
    # Erode mask by 3 pixels to get inner ring
    erode_3px_kernel = np.ones((3, 3), np.uint8)
    inner_mask_3px = cv2.erode(mask, erode_3px_kernel, iterations=3)
    ring_3px = cv2.subtract(mask, inner_mask_3px)
    
    # Get pixels in the 3px depth ring
    ring_3px_pixels = img[ring_3px > 0]
    
    if len(ring_3px_pixels) > 0:
        avg_color_3px_bgr = ring_3px_pixels.mean(axis=0)
        border_color_3px = {
            'r': int(avg_color_3px_bgr[2]),
            'g': int(avg_color_3px_bgr[1]),
            'b': int(avg_color_3px_bgr[0]),
            'hex': f"#{int(avg_color_3px_bgr[2]):02x}{int(avg_color_3px_bgr[1]):02x}{int(avg_color_3px_bgr[0]):02x}"
        }
    else:
        border_color_3px = avg_color_rgb.copy()
    
    # Detect border thickness by analyzing color gradient from edge inward
    # Sample at multiple depths and find where color significantly changes
    border_thickness = 0
    if circle_info and len(outer_pixels) > 0:
        max_sample_depth = 20  # Maximum depth to sample
        color_threshold = 15  # RGB color difference threshold
        
        reference_color = avg_color_bgr
        
        for depth in range(1, max_sample_depth + 1):
            # Create ring at this depth
            inner_mask_at_depth = cv2.erode(mask, erode_3px_kernel, iterations=depth)
            ring_at_depth = cv2.subtract(mask, inner_mask_at_depth)
            
            if ring_at_depth.sum() == 0:
                break
            
            depth_pixels = img[ring_at_depth > 0]
            if len(depth_pixels) > 0:
                depth_color = depth_pixels.mean(axis=0)
                
                # Calculate color difference
                color_diff = np.sqrt(np.sum((depth_color - reference_color) ** 2))
                
                if color_diff > color_threshold:
                    border_thickness = depth
                    break
        
        # If no significant change detected, assume thin border
        if border_thickness == 0:
            border_thickness = 3
    
    # Convert image to RGBA
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # Apply mask to alpha channel
    img_rgba[:, :, 3] = mask
    
    # Create metadata with shape classification
    ellipse = circle_info.get('ellipse') if circle_info else None
    circularity = round(circle_info['circularity'], 3) if circle_info else 0
    aspect_ratio = round(ellipse['aspect_ratio'], 3) if ellipse else 1.0
    radial_variance = round(circle_info.get('radial_variance', 1.0), 4) if circle_info else 1.0
    is_radially_circular = circle_info.get('is_radially_circular', False) if circle_info else False
    
    # Classify shape using radial variance (more robust than perimeter-based circularity)
    # Radial variance < 3% = circle (uniform radius in all directions)
    # Radial variance >= 3% = rounded square (flat sides cause variance)
    shape_type = 'unknown'
    if is_radially_circular and 0.97 <= aspect_ratio <= 1.03:
        shape_type = 'circle'
    elif radial_variance < 0.06 and 0.97 <= aspect_ratio <= 1.03:
        # Slightly higher variance but still relatively circular
        shape_type = 'circle'
    elif radial_variance >= 0.06 and 0.97 <= aspect_ratio <= 1.03:
        shape_type = 'rounded_square'
    elif circularity < 0.80 and 0.97 <= aspect_ratio <= 1.03:
        shape_type = 'square'
    elif radial_variance >= 0.06 and aspect_ratio > 1.03:
        shape_type = 'rounded_rect'
    else:
        shape_type = 'rect'
    
    # Save result
    cv2.imwrite(str(output_path), img_rgba)
    print(f"Processed: {image_path.name} -> {output_path.name} [{shape_type}]")
    
    # Build shape info
    shape_info = {
        'type': shape_type,
        'aspect_ratio': aspect_ratio,
        'major_axis': round(ellipse['major_axis'], 1) if ellipse else 0,
        'minor_axis': round(ellipse['minor_axis'], 1) if ellipse else 0,
        'ellipse_angle': round(ellipse['angle'], 1) if ellipse else 0,
        'radial_variance': radial_variance
    }
    
    # Add circularity for reference (but not used in classification anymore)
    if shape_type in ['circle', 'rounded_square']:
        shape_info['circularity'] = circularity
    
    metadata = {
        'filename': image_path.name,
        'xsize': width,
        'ysize': height,
        'xpos': round(circle_info['x'], 1) if circle_info else 0,
        'ypos': round(circle_info['y'], 1) if circle_info else 0,
        'diameter': round(circle_info['radius'] * 2, 1) if circle_info else 0,
        'shape': shape_info,
        'border_color': avg_color_rgb,
        'border_color_3px': border_color_3px,
        'border_thickness': border_thickness
    }
    
    return metadata


def process_all_images(input_folder="img/input", output_subfolder="img/output"):
    """
    Process all images in the input folder and save to output folder.
    Supports both direct images and images in subfolders.
    
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
    
    # Find all images - only in subfolders, not root
    image_files = []
    
    # Process only subfolders in input folder
    for item in input_path.iterdir():
        if item.is_dir():
            # Process subfolder
            subfolder_output = output_path / item.name
            subfolder_output.mkdir(parents=True, exist_ok=True)
            for sub_item in item.iterdir():
                if sub_item.is_file() and sub_item.suffix.lower() in image_extensions:
                    image_files.append((sub_item, subfolder_output))
    
    if not image_files:
        print(f"No images found in {input_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print("-" * 50)
    
    # Process each image and collect metadata
    image_metadata = []
    success_count = 0
    for image_file, output_folder in image_files:
        # Create output filename with .png extension (to support transparency)
        output_file = output_folder / f"{image_file.stem}.png"
        
        metadata = remove_background(image_file, output_file)
        if metadata:
            # Add subfolder info to metadata if in a subfolder
            if output_folder != output_path:
                metadata['subfolder'] = output_folder.name
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
                "x_margin": 0.3,
                "y_margin": 0.3,
                "padding": 0.15,
                "token_size": 1.0,
                "ppi": [300, 600],
                "brightness_adjustment": "1-0-1-1"
            },
            "tokens_quantity_list": [
                {
                    "filename": meta["filename"], 
                    "quantity": 1,
                    **({'subfolder': meta['subfolder']} if 'subfolder' in meta else {})
                }
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

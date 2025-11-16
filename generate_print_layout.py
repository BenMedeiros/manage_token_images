"""
Print Layout Generator

This script processes the print_format_template.json, calculates grid layout,
and generates a printable page layout with tokens arranged in a grid.
"""

import json
import cv2
import numpy as np
from pathlib import Path


def calculate_tokens_per_page(settings):
    """
    Calculate how many tokens fit per row and column based on print settings.
    
    Args:
        settings: Dictionary containing print settings
        
    Returns:
        tuple: (tokens_per_row, tokens_per_col)
    """
    print_width = settings['print_width']
    print_height = settings['print_height']
    x_margin = settings['x_margin']
    y_margin = settings['y_margin']
    x_spacer = settings['x_spacer']
    y_spacer = settings['y_spacer']
    token_size = settings['token_size']
    
    # Calculate tokens per row
    available_width = print_width - 2 * x_margin
    tokens_per_row = int((available_width + x_spacer) / (token_size + x_spacer))
    
    # Calculate tokens per column
    available_height = print_height - 2 * y_margin
    tokens_per_col = int((available_height + y_spacer) / (token_size + y_spacer))
    
    return tokens_per_row, tokens_per_col


def update_template_calculations(template_path):
    """
    Update the template with calculated values.
    
    Args:
        template_path: Path to print_format_template.json
    """
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    settings = template['settings']
    
    # Calculate tokens per page
    tokens_per_row, tokens_per_col = calculate_tokens_per_page(settings)
    
    # Calculate wasted space
    token_size = settings['token_size']
    x_margin = settings['x_margin']
    y_margin = settings['y_margin']
    x_spacer = settings['x_spacer']
    y_spacer = settings['y_spacer']
    
    used_width = 2 * x_margin + tokens_per_row * token_size + (tokens_per_row - 1) * x_spacer
    wasted_x = settings['print_width'] - used_width
    
    used_height = 2 * y_margin + tokens_per_col * token_size + (tokens_per_col - 1) * y_spacer
    wasted_y = settings['print_height'] - used_height
    
    # Add calculated values
    if 'calculated' not in settings:
        settings['calculated'] = {}
    
    settings['calculated']['tokens_per_row'] = tokens_per_row
    settings['calculated']['tokens_per_col'] = tokens_per_col
    settings['calculated']['tokens_per_page'] = tokens_per_row * tokens_per_col
    settings['calculated']['wasted_x'] = round(wasted_x, 3)
    settings['calculated']['wasted_y'] = round(wasted_y, 3)
    
    # Save updated template
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Template calculations updated")
    print(f"Tokens per row: {tokens_per_row}")
    print(f"Tokens per column: {tokens_per_col}")
    print(f"Total tokens per page: {tokens_per_row * tokens_per_col}")
    
    total_tokens = sum(item['quantity'] for item in template['tokens_quantity_list'])
    tokens_per_page = tokens_per_row * tokens_per_col
    pages_needed = (total_tokens + tokens_per_page - 1) // tokens_per_page
    
    print(f"Total tokens to print: {total_tokens}")
    print(f"Pages needed: {pages_needed}")
    print("-" * 50)
    
    return template


def create_print_layout(template, output_base_path, metadata_path):
    """
    Create printable layout images from the template, generating multiple pages if needed.
    
    Args:
        template: Template dictionary
        output_base_path: Base path for output files (without .png extension)
        metadata_path: Path to image_metadata.json file
        
    Returns:
        List of output file paths created
    """
    settings = template['settings']
    tokens_list = template['tokens_quantity_list']
    
    # Load metadata for border colors
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    
    # Create lookup dict for metadata by filename
    metadata_dict = {meta['filename']: meta for meta in metadata_list}
    
    # Get settings
    print_width = settings['print_width']
    print_height = settings['print_height']
    x_margin = settings['x_margin']
    y_margin = settings['y_margin']
    x_spacer = settings['x_spacer']
    y_spacer = settings['y_spacer']
    token_size = settings['token_size']
    ppi = settings['ppi']
    
    # Calculate image dimensions in pixels
    page_width_px = int(print_width * ppi)
    page_height_px = int(print_height * ppi)
    token_size_px = int(token_size * ppi)
    x_margin_px = int(x_margin * ppi)
    y_margin_px = int(y_margin * ppi)
    x_spacer_px = int(x_spacer * ppi)
    y_spacer_px = int(y_spacer * ppi)
    
    # Get calculated values
    if 'calculated' in settings:
        tokens_per_row = settings['calculated']['tokens_per_row']
        tokens_per_col = settings['calculated']['tokens_per_col']
    else:
        # Calculate if not present
        available_width = print_width - 2 * x_margin
        tokens_per_row = int((available_width + x_spacer) / (token_size + x_spacer))
        available_height = print_height - 2 * y_margin
        tokens_per_col = int((available_height + y_spacer) / (token_size + y_spacer))
    
    tokens_per_page = tokens_per_row * tokens_per_col
    
    # Get img folder path (assume template is in img folder)
    img_folder = Path(output_base_path).parent / "output"
    
    # Build list of tokens to place (respecting quantities)
    tokens_to_place = []
    for item in tokens_list:
        filename = item['filename']
        quantity = item['quantity']
        for _ in range(quantity):
            tokens_to_place.append(filename)
    
    total_tokens = len(tokens_to_place)
    pages_needed = (total_tokens + tokens_per_page - 1) // tokens_per_page
    
    print(f"Total tokens to place: {total_tokens}")
    print(f"Page dimensions: {page_width_px}x{page_height_px} pixels")
    print(f"Token size: {token_size_px}x{token_size_px} pixels")
    print(f"Grid: {tokens_per_row} columns x {tokens_per_col} rows")
    print(f"Tokens per page: {tokens_per_page}")
    print(f"Pages needed: {pages_needed}")
    print("-" * 50)
    
    output_files = []
    token_index = 0
    
    # Generate each page
    for page_num in range(pages_needed):
        # Create blank white page
        page = np.ones((page_height_px, page_width_px, 4), dtype=np.uint8) * 255
        
        # Draw grid lines BEFORE placing tokens
        gray = (128, 128, 128)  # Gray color for grid lines
        line_thickness = 1
        
        # Draw margin lines (page boundaries)
        cv2.rectangle(page, 
                     (x_margin_px, y_margin_px),
                     (page_width_px - x_margin_px, page_height_px - y_margin_px),
                     gray, line_thickness)
        
        # Draw vertical grid lines
        for col in range(tokens_per_row + 1):
            # Left edge of each token column
            x_left = x_margin_px + col * (token_size_px + x_spacer_px)
            cv2.line(page, (x_left, y_margin_px), (x_left, page_height_px - y_margin_px), gray, line_thickness)
            
            # Right edge (creates spacer line) - only draw if not the last column
            if col < tokens_per_row:
                x_right = x_left + token_size_px
                cv2.line(page, (x_right, y_margin_px), (x_right, page_height_px - y_margin_px), gray, line_thickness)
                
                # Center line of each token (midpoint)
                x_center = x_left + token_size_px // 2
                cv2.line(page, (x_center, y_margin_px), (x_center, page_height_px - y_margin_px), gray, line_thickness)
        
        # Draw horizontal grid lines
        for row in range(tokens_per_col + 1):
            # Top edge of each token row
            y_top = y_margin_px + row * (token_size_px + y_spacer_px)
            cv2.line(page, (x_margin_px, y_top), (page_width_px - x_margin_px, y_top), gray, line_thickness)
            
            # Bottom edge (creates spacer line) - only draw if not the last row
            if row < tokens_per_col:
                y_bottom = y_top + token_size_px
                cv2.line(page, (x_margin_px, y_bottom), (page_width_px - x_margin_px, y_bottom), gray, line_thickness)
                
                # Center line of each token (midpoint)
                y_center = y_top + token_size_px // 2
                cv2.line(page, (x_margin_px, y_center), (page_width_px - x_margin_px, y_center), gray, line_thickness)
        
        tokens_on_this_page = 0
        
        # Place tokens on this page
        for row in range(tokens_per_col):
            for col in range(tokens_per_row):
                if token_index >= total_tokens:
                    break
                
                filename = tokens_to_place[token_index]
                token_path = img_folder / filename
                
                # Calculate position for this token
                x_pos = x_margin_px + col * (token_size_px + x_spacer_px)
                y_pos = y_margin_px + row * (token_size_px + y_spacer_px)
                
                # Fill background with border color from metadata
                if filename in metadata_dict:
                    border_color = metadata_dict[filename]['border_color']
                    # OpenCV uses BGR format
                    bg_color = (border_color['b'], border_color['g'], border_color['r'], 255)
                    page[y_pos:y_pos+token_size_px, x_pos:x_pos+token_size_px] = bg_color
                
                # Load token image
                token_img = cv2.imread(str(token_path), cv2.IMREAD_UNCHANGED)
                
                if token_img is None:
                    print(f"Warning: Could not load {filename}")
                    token_index += 1
                    continue
                
                # Crop to the bounding box of non-transparent pixels
                if token_img.shape[2] == 4:  # Has alpha channel
                    # Find non-transparent pixels
                    alpha_mask = token_img[:, :, 3] > 0
                    rows = np.any(alpha_mask, axis=1)
                    cols = np.any(alpha_mask, axis=0)
                    
                    if rows.any() and cols.any():
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        
                        # Crop to content
                        token_img = token_img[y_min:y_max+1, x_min:x_max+1]
                
                # Resize token to target size using highest quality interpolation
                token_resized = cv2.resize(token_img, (token_size_px, token_size_px), 
                                          interpolation=cv2.INTER_LANCZOS4)
                
                # Place token on page using alpha blending
                if token_resized.shape[2] == 4:  # Has alpha channel
                    # Extract alpha channel
                    alpha = token_resized[:, :, 3] / 255.0
                    
                    # Blend token onto page
                    for c in range(3):  # RGB channels
                        page[y_pos:y_pos+token_size_px, x_pos:x_pos+token_size_px, c] = \
                            (alpha * token_resized[:, :, c] + 
                             (1 - alpha) * page[y_pos:y_pos+token_size_px, x_pos:x_pos+token_size_px, c])
                    
                    # Update alpha channel (make opaque where token is placed)
                    page[y_pos:y_pos+token_size_px, x_pos:x_pos+token_size_px, 3] = 255
                else:
                    # No alpha channel, just copy
                    page[y_pos:y_pos+token_size_px, x_pos:x_pos+token_size_px, :3] = token_resized
                
                token_index += 1
                tokens_on_this_page += 1
            
            if token_index >= total_tokens:
                break
        
        # Create output filename for this page
        if pages_needed > 1:
            output_path = Path(str(output_base_path).replace('.png', '') + f"_page{page_num+1}.png")
        else:
            output_path = Path(str(output_base_path))
        
        # Save page
        cv2.imwrite(str(output_path), page)
        
        # Add PPI metadata to PNG
        add_ppi_metadata(output_path, ppi)
        
        output_files.append(output_path)
        print(f"Page {page_num+1}: Placed {tokens_on_this_page} tokens -> {output_path.name}")
    
    print("-" * 50)
    print(f"Generated {len(output_files)} page(s)")
    
    return output_files


def add_ppi_metadata(output_path, ppi):
    """Add PPI metadata to a PNG file."""
    try:
        with open(output_path, 'rb') as f:
            png_data = bytearray(f.read())
        
        # Check if pHYs chunk already exists
        if b'pHYs' not in png_data:
            # Find IDAT chunk position
            idat_pos = png_data.find(b'IDAT') - 4
            if idat_pos > 0:
                # Create pHYs chunk
                ppm = int(ppi / 0.0254)  # Convert PPI to pixels per meter
                phys_data = ppm.to_bytes(4, 'big') + ppm.to_bytes(4, 'big') + b'\x01'
                phys_length = len(phys_data).to_bytes(4, 'big')
                phys_chunk = phys_length + b'pHYs' + phys_data
                
                # Calculate CRC
                import zlib
                crc = zlib.crc32(b'pHYs' + phys_data) & 0xffffffff
                phys_chunk += crc.to_bytes(4, 'big')
                
                # Insert pHYs chunk
                png_data = png_data[:idat_pos] + phys_chunk + png_data[idat_pos:]
                
                # Write modified PNG
                with open(output_path, 'wb') as f:
                    f.write(png_data)
    except:
        pass


if __name__ == "__main__":
    # Get paths
    script_dir = Path(__file__).parent
    template_path = script_dir / "img" / "print_format_template.json"
    metadata_path = script_dir / "img" / "image_metadata.json"
    
    if not template_path.exists():
        print(f"Error: Template not found at {template_path}")
        print("Please run process_tokens.py first to generate the template.")
    elif not metadata_path.exists():
        print(f"Error: Metadata not found at {metadata_path}")
        print("Please run process_tokens.py first to generate the metadata.")
    else:
        # Update template with calculations
        template = update_template_calculations(template_path)
        
        settings = template['settings']
        width = settings['print_width']
        height = settings['print_height']
        ppi_setting = settings['ppi']
        
        # Handle ppi as either a single value or array
        ppi_values = ppi_setting if isinstance(ppi_setting, list) else [ppi_setting]
        
        # Generate layout for each PPI value
        for ppi in ppi_values:
            print(f"\n{'='*50}")
            print(f"Generating layout at {ppi} PPI")
            print(f"{'='*50}")
            
            # Update settings with current PPI
            settings['ppi'] = ppi
            
            # Create output filename with dimensions and PPI (replace decimal points)
            width_str = str(width).replace('.', '_')
            height_str = str(height).replace('.', '_')
            output_filename = f"output_{width_str}x{height_str}_{ppi}ppi.png"
            output_path = script_dir / "img" / output_filename
            
            # Generate the layout
            create_print_layout(template, output_path, metadata_path)

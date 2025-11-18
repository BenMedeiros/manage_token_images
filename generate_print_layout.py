"""
Print Layout Generator

This script processes the print_format_template.json, calculates grid layout,
and generates a printable page layout with tokens arranged in a grid.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import re
import itertools


def expand_brightness_string(brightness_str):
    """
    Expand brightness string with array notation into multiple strings.
    
    Examples:
        "[1,1.5,2]-0-1-1" -> ["1-0-1-1", "1.5-0-1-1", "2-0-1-1"]
        "1.5-0-[1,1.5]-1" -> ["1.5-0-1-1", "1.5-0-1.5-1"]
        "[1,2]-0-1-[1,2]" -> ["1-0-1-1", "1-0-1-2", "2-0-1-1", "2-0-1-2"]
    
    Args:
        brightness_str: String with optional array notation
        
    Returns:
        List of expanded brightness strings
    """
    if '[' not in brightness_str:
        return [brightness_str]
    
    # Find all array patterns [x,y,z]
    array_pattern = r'\[([^\]]+)\]'
    matches = list(re.finditer(array_pattern, brightness_str))
    
    if not matches:
        return [brightness_str]
    
    # Extract arrays
    arrays = []
    for match in matches:
        values_str = match.group(1)
        values = [v.strip() for v in values_str.split(',')]
        arrays.append(values)
    
    # Generate all combinations using cartesian product
    combinations = list(itertools.product(*arrays))
    
    # Replace arrays with values for each combination
    result = []
    for combo in combinations:
        output = brightness_str
        for i, match in enumerate(reversed(matches)):  # Reverse to maintain indices
            output = output[:match.start()] + combo[len(combo)-1-i] + output[match.end():]
        result.append(output)
    
    return result


def apply_brightness_adjustment(img, settings, token_info):
    """
    Apply brightness adjustments to an image for photo paper printing.
    
    Adjustments preserve blacks while brightening midtones and highlights.
    Can be set at template level (settings) and overridden per-token (token_info).
    
    Args:
        img: Image array (BGR or BGRA)
        settings: Template settings dict
        token_info: Individual token dict
        
    Returns:
        Adjusted image array
    """
    # Get brightness adjustment settings (token overrides template)
    brightness = token_info.get('brightness_adjustment') or settings.get('brightness_adjustment')
    
    if not brightness:
        return img
    
    # Expand array notation if present and take first result (for template-level settings)
    if isinstance(brightness, str) and '[' in brightness:
        expanded = expand_brightness_string(brightness)
        if expanded:
            brightness = expanded[0]  # Use first expanded value
        else:
            return img
    
    # Parse brightness adjustment (string format "g-sl-mb-hb" or dict)
    if isinstance(brightness, str):
        parts = brightness.split('-')
        if len(parts) != 4:
            print(f"Warning: Invalid brightness string '{brightness}', expected format 'g-sl-mb-hb'")
            return img
        try:
            gamma = float(parts[0])
            shadow_lift = float(parts[1])
            midtone_boost = float(parts[2])
            highlight_boost = float(parts[3])
        except ValueError:
            print(f"Warning: Invalid brightness values in '{brightness}'")
            return img
    elif isinstance(brightness, dict):
        # Extract adjustment parameters with defaults
        gamma = brightness.get('gamma', 1.0)
        shadow_lift = brightness.get('shadow_lift', 0.0)
        midtone_boost = brightness.get('midtone_boost', 1.0)
        highlight_boost = brightness.get('highlight_boost', 1.0)
    else:
        return img
    
    # Separate alpha channel if present
    has_alpha = img.shape[2] == 4
    if has_alpha:
        alpha = img[:, :, 3].copy()
        img_rgb = img[:, :, :3].copy()
    else:
        img_rgb = img.copy()
    
    # Convert to float [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # Apply gamma correction (affects midtones most)
    if gamma != 1.0:
        img_float = np.power(img_float, 1.0 / gamma)
    
    # Apply shadow/midtone/highlight adjustments
    # Shadows: 0-0.33, Midtones: 0.33-0.67, Highlights: 0.67-1.0
    if shadow_lift != 0.0 or midtone_boost != 1.0 or highlight_boost != 1.0:
        # Create masks for each tonal range
        luminance = 0.299 * img_float[:, :, 2] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 0]
        
        # Shadow mask (peaks at 0, fades by 0.33)
        shadow_mask = np.clip(1.0 - luminance / 0.33, 0, 1)
        
        # Midtone mask (peaks at 0.5, fades at 0.33 and 0.67)
        midtone_mask = np.zeros_like(luminance)
        low_mid = luminance < 0.5
        midtone_mask[low_mid] = (luminance[low_mid] - 0.33) / 0.17
        midtone_mask[~low_mid] = (0.67 - luminance[~low_mid]) / 0.17
        midtone_mask = np.clip(midtone_mask, 0, 1)
        
        # Highlight mask (peaks at 1.0, fades by 0.67)
        highlight_mask = np.clip((luminance - 0.67) / 0.33, 0, 1)
        
        # Apply adjustments per channel
        for c in range(3):
            if shadow_lift != 0.0:
                img_float[:, :, c] += shadow_mask * shadow_lift
            if midtone_boost != 1.0:
                img_float[:, :, c] = img_float[:, :, c] * (1 + midtone_mask * (midtone_boost - 1.0))
            if highlight_boost != 1.0:
                img_float[:, :, c] = img_float[:, :, c] * (1 + highlight_mask * (highlight_boost - 1.0))
    
    # Clip and convert back to uint8
    img_float = np.clip(img_float, 0, 1)
    img_adjusted = (img_float * 255).astype(np.uint8)
    
    # Restore alpha channel if present
    if has_alpha:
        img_result = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        img_result[:, :, :3] = img_adjusted
        img_result[:, :, 3] = alpha
        return img_result
    
    return img_adjusted


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
    padding = settings.get('padding', 0.0)
    token_size = settings['token_size']

    # Calculate tokens per row
    available_width = print_width - 2 * x_margin
    tokens_per_row = int(available_width / (token_size + 2 * padding))

    # Calculate tokens per column
    available_height = print_height - 2 * y_margin
    tokens_per_col = int(available_height / (token_size + 2 * padding))

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
    padding = settings.get('padding', 0.0)

    used_width = 2 * x_margin + tokens_per_row * (token_size + 2 * padding)
    wasted_x = settings['print_width'] - used_width

    used_height = 2 * y_margin + tokens_per_col * (token_size + 2 * padding)
    wasted_y = settings['print_height'] - used_height
    
    # Calculate token counts (handle brightness_adjustment arrays and expansion)
    total_tokens = 0
    for item in template['tokens_quantity_list']:
        brightness_adj = item.get('brightness_adjustment')
        if isinstance(brightness_adj, list):
            # Count expanded tokens from array
            for brightness_config in brightness_adj:
                if isinstance(brightness_config, str):
                    expanded = expand_brightness_string(brightness_config)
                    total_tokens += len(expanded)
                else:
                    total_tokens += 1
        else:
            if brightness_adj and isinstance(brightness_adj, str):
                expanded = expand_brightness_string(brightness_adj)
                total_tokens += len(expanded)
            else:
                total_tokens += item['quantity']
    
    tokens_per_page = tokens_per_row * tokens_per_col
    pages_needed = (total_tokens + tokens_per_page - 1) // tokens_per_page
    
    # Calculate wasted token slots on last page
    total_slots = pages_needed * tokens_per_page
    wasted_tokens = total_slots - total_tokens
    
    # Add calculated values
    if 'calculated' not in settings:
        settings['calculated'] = {}
    
    settings['calculated']['tokens_per_row'] = tokens_per_row
    settings['calculated']['tokens_per_col'] = tokens_per_col
    settings['calculated']['tokens_per_page'] = tokens_per_row * tokens_per_col
    settings['calculated']['total_tokens'] = total_tokens
    settings['calculated']['wasted_tokens'] = wasted_tokens
    settings['calculated']['wasted_x'] = round(wasted_x, 3)
    settings['calculated']['wasted_y'] = round(wasted_y, 3)
    
    # Save updated template
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Template calculations updated")
    print(f"Tokens per row: {tokens_per_row}")
    print(f"Tokens per column: {tokens_per_col}")
    print(f"Total tokens per page: {tokens_per_row * tokens_per_col}")
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
    
    # Create lookup dict for metadata by (subfolder, filename) to handle duplicates
    metadata_dict = {(meta.get('subfolder'), meta['filename']): meta for meta in metadata_list}
    
    # Get settings
    print_width = settings['print_width']
    print_height = settings['print_height']
    x_margin = settings['x_margin']
    y_margin = settings['y_margin']
    padding = settings.get('padding', 0.0)
    token_size = settings['token_size']
    ppi = settings['ppi']
    
    # Calculate image dimensions in pixels
    page_width_px = int(print_width * ppi)
    page_height_px = int(print_height * ppi)
    token_size_px = int(token_size * ppi)
    x_margin_px = int(x_margin * ppi)
    y_margin_px = int(y_margin * ppi)
    padding_px = int(padding * ppi)
    
    # Get calculated values
    if 'calculated' in settings:
        tokens_per_row = settings['calculated']['tokens_per_row']
        tokens_per_col = settings['calculated']['tokens_per_col']
    else:
        # Calculate if not present
        available_width = print_width - 2 * x_margin
        tokens_per_row = int(available_width / (token_size + 2 * padding))
        available_height = print_height - 2 * y_margin
        tokens_per_col = int(available_height / (token_size + 2 * padding))
    
    tokens_per_page = tokens_per_row * tokens_per_col
    
    # Get img folder path - tokens are always in img/output regardless of where template is
    script_dir = Path(__file__).parent
    img_folder = script_dir / "img" / "output"
    
    # Build list of tokens to place (respecting quantities and subfolders)
    tokens_to_place = []
    for item in tokens_list:
        filename = item['filename']
        subfolder = item.get('subfolder')  # Optional subfolder field
        quantity = item['quantity']
        
        # Handle brightness_adjustment as array or single value
        brightness_adj = item.get('brightness_adjustment')
        if isinstance(brightness_adj, list):
            # Array of brightness configs - create one token per config
            for brightness_config in brightness_adj:
                # Expand array notation in brightness string
                if isinstance(brightness_config, str):
                    expanded = expand_brightness_string(brightness_config)
                    for expanded_str in expanded:
                        tokens_to_place.append({
                            'filename': filename,
                            'subfolder': subfolder,
                            'brightness_adjustment': expanded_str
                        })
                else:
                    tokens_to_place.append({
                        'filename': filename,
                        'subfolder': subfolder,
                        'brightness_adjustment': brightness_config
                    })
        else:
            # Single brightness config or none - check template default
            # Get template-level brightness setting
            template_brightness = settings.get('brightness_adjustment')
            
            # Determine effective brightness setting
            effective_brightness = brightness_adj or template_brightness
            
            if effective_brightness:
                # Handle array of brightness strings at template level
                if isinstance(effective_brightness, list):
                    # Array of brightness configs - expand each and combine
                    all_expanded = []
                    for brightness_config in effective_brightness:
                        if isinstance(brightness_config, str):
                            expanded = expand_brightness_string(brightness_config)
                            all_expanded.extend(expanded)
                        else:
                            all_expanded.append(brightness_config)
                    
                    # Create quantity copies per expansion
                    for _ in range(quantity):
                        for expanded_str in all_expanded:
                            tokens_to_place.append({
                                'filename': filename,
                                'subfolder': subfolder,
                                'brightness_adjustment': expanded_str
                            })
                # Handle single brightness string or dict
                elif isinstance(effective_brightness, str):
                    expanded = expand_brightness_string(effective_brightness)
                    if len(expanded) > 1:
                        # Multiple expanded values - create quantity copies per expansion
                        for _ in range(quantity):
                            for expanded_str in expanded:
                                tokens_to_place.append({
                                    'filename': filename,
                                    'subfolder': subfolder,
                                    'brightness_adjustment': expanded_str
                                })
                    else:
                        # Single value after expansion
                        for _ in range(quantity):
                            tokens_to_place.append({
                                'filename': filename,
                                'subfolder': subfolder,
                                'brightness_adjustment': expanded[0]
                            })
                else:
                    # Dict format or other
                    for _ in range(quantity):
                        tokens_to_place.append({
                            'filename': filename,
                            'subfolder': subfolder,
                            'brightness_adjustment': effective_brightness
                        })
            else:
                # No brightness adjustment
                for _ in range(quantity):
                    tokens_to_place.append({
                        'filename': filename,
                        'subfolder': subfolder
                    })
    
    total_tokens = len(tokens_to_place)
    pages_needed = (total_tokens + tokens_per_page - 1) // tokens_per_page
    
    print(f"Total tokens to place: {total_tokens}")
    print(f"Page dimensions: {page_width_px}x{page_height_px} pixels")
    print(f"Token size: {token_size_px}x{token_size_px} pixels")
    print(f"Grid: {tokens_per_row} columns x {tokens_per_col} rows")
    print(f"Tokens per page: {tokens_per_page}")
    print(f"Pages needed: {pages_needed}")
    print("-" * 50)
    
    # Calculate centering offsets for grid
    # Total space used by grid
    grid_width = tokens_per_row * (token_size_px + 2 * padding_px)
    grid_height = tokens_per_col * (token_size_px + 2 * padding_px)
    
    # Available space within margins
    available_width = page_width_px - 2 * x_margin_px
    available_height = page_height_px - 2 * y_margin_px
    
    # Calculate offsets to center the grid
    x_offset = (available_width - grid_width) // 2
    y_offset = (available_height - grid_height) // 2
    
    output_files = []
    token_index = 0
    
    # Generate each page
    for page_num in range(pages_needed):
        # Create blank white page (BGRA format)
        page = np.ones((page_height_px, page_width_px, 4), dtype=np.uint8) * 255
        gray_bgr = (128, 128, 128, 255)
        line_thickness = 1
        # Draw all 4 margin lines (page boundaries)
        # Top
        cv2.line(page, (0, y_margin_px), (page_width_px, y_margin_px), gray_bgr, line_thickness)
        # Bottom
        cv2.line(page, (0, page_height_px - y_margin_px), (page_width_px, page_height_px - y_margin_px), gray_bgr, line_thickness)
        # Left
        cv2.line(page, (x_margin_px, 0), (x_margin_px, page_height_px), gray_bgr, line_thickness)
        # Right
        cv2.line(page, (page_width_px - x_margin_px, 0), (page_width_px - x_margin_px, page_height_px), gray_bgr, line_thickness)
        
        # Draw grid boundary lines (showing actual grid area with centering)
        if x_offset > 0 or y_offset > 0:
            grid_color = (180, 180, 180, 255)  # Lighter gray for grid boundaries
            grid_left = x_margin_px + x_offset
            grid_right = grid_left + grid_width
            grid_top = y_margin_px + y_offset
            grid_bottom = grid_top + grid_height
            
            # Horizontal lines (top and bottom of grid, extending full width)
            cv2.line(page, (0, grid_top), (page_width_px, grid_top), grid_color, line_thickness)
            cv2.line(page, (0, grid_bottom), (page_width_px, grid_bottom), grid_color, line_thickness)
            # Vertical lines (left and right of grid, extending full height)
            cv2.line(page, (grid_left, 0), (grid_left, page_height_px), grid_color, line_thickness)
            cv2.line(page, (grid_right, 0), (grid_right, page_height_px), grid_color, line_thickness)
        
        # Draw measurement tick marks every 1 inch
        tick_color = (100, 100, 100, 255)  # Dark gray for tick marks
        tick_length = y_margin_px  # Margin height for vertical ticks
        inch_interval = ppi  # 1 inch in pixels
        
        # Horizontal tick marks along top and bottom edges (within margins)
        x_pos = inch_interval
        while x_pos < page_width_px:
            # Top edge ticks (extending down into top margin)
            cv2.line(page, (x_pos, 0), (x_pos, y_margin_px), tick_color, line_thickness)
            # Bottom edge ticks (extending up into bottom margin)
            cv2.line(page, (x_pos, page_height_px - y_margin_px), (x_pos, page_height_px), tick_color, line_thickness)
            x_pos += inch_interval
        
        # Vertical tick marks along left and right edges (within margins)
        y_pos = inch_interval
        while y_pos < page_height_px:
            # Left edge ticks (extending right into left margin)
            cv2.line(page, (0, y_pos), (x_margin_px, y_pos), tick_color, line_thickness)
            # Right edge ticks (extending left into right margin)
            cv2.line(page, (page_width_px - x_margin_px, y_pos), (page_width_px, y_pos), tick_color, line_thickness)
            y_pos += inch_interval
        
        tokens_on_this_page = 0
        
        # Draw token padding borders
        temp_token_index = token_index
        for row in range(tokens_per_col):
            for col in range(tokens_per_row):
                if temp_token_index >= total_tokens:
                    break
                token_info = tokens_to_place[temp_token_index]
                filename = token_info['filename']
                subfolder = token_info['subfolder']
                # Per-token padding override
                token_padding = padding
                if 'padding' in token_info:
                    token_padding = token_info['padding']
                token_padding_px = int(token_padding * ppi)
                # Calculate center position for this token's slot (with centering offset)
                slot_x = x_margin_px + x_offset + col * (token_size_px + 2 * padding_px)
                slot_y = y_margin_px + y_offset + row * (token_size_px + 2 * padding_px)
                center_x = slot_x + token_size_px // 2 + token_padding_px
                center_y = slot_y + token_size_px // 2 + token_padding_px
                # Draw border: circle or rounded square based on token shape
                metadata_key = (subfolder, filename)
                if metadata_key in metadata_dict:
                    meta = metadata_dict[metadata_key]
                    shape_info = meta.get('shape', {})
                    shape_type = shape_info.get('type', 'unknown')
                    
                    if shape_type == 'circle':
                        # Draw circle at padded radius
                        radius = (token_size_px // 2) + token_padding_px
                        cv2.circle(page, (center_x, center_y), radius, gray_bgr, line_thickness)
                    else:
                        # Draw rounded rectangle at padded size
                        rect_size = token_size_px + 2 * token_padding_px
                        rect_x = slot_x
                        rect_y = slot_y
                        # Calculate corner radius based on original token's roundedness
                        # For rounded squares, use about 15-20% of size as radius
                        corner_radius = int(0.18 * rect_size)
                        
                        # Draw rounded rectangle using polylines
                        pts = []
                        # Top-left corner
                        pts.append([rect_x + corner_radius, rect_y])
                        # Top-right corner
                        pts.append([rect_x + rect_size - corner_radius, rect_y])
                        for angle in range(0, 91, 10):
                            rad = np.radians(angle - 90)
                            x = int(rect_x + rect_size - corner_radius + corner_radius * np.cos(rad))
                            y = int(rect_y + corner_radius + corner_radius * np.sin(rad))
                            pts.append([x, y])
                        # Right edge
                        pts.append([rect_x + rect_size, rect_y + corner_radius])
                        pts.append([rect_x + rect_size, rect_y + rect_size - corner_radius])
                        # Bottom-right corner
                        for angle in range(0, 91, 10):
                            rad = np.radians(angle)
                            x = int(rect_x + rect_size - corner_radius + corner_radius * np.cos(rad))
                            y = int(rect_y + rect_size - corner_radius + corner_radius * np.sin(rad))
                            pts.append([x, y])
                        # Bottom edge
                        pts.append([rect_x + rect_size - corner_radius, rect_y + rect_size])
                        pts.append([rect_x + corner_radius, rect_y + rect_size])
                        # Bottom-left corner
                        for angle in range(0, 91, 10):
                            rad = np.radians(angle + 90)
                            x = int(rect_x + corner_radius + corner_radius * np.cos(rad))
                            y = int(rect_y + rect_size - corner_radius + corner_radius * np.sin(rad))
                            pts.append([x, y])
                        # Left edge
                        pts.append([rect_x, rect_y + rect_size - corner_radius])
                        pts.append([rect_x, rect_y + corner_radius])
                        # Top-left corner
                        for angle in range(0, 91, 10):
                            rad = np.radians(angle + 180)
                            x = int(rect_x + corner_radius + corner_radius * np.cos(rad))
                            y = int(rect_y + corner_radius + corner_radius * np.sin(rad))
                            pts.append([x, y])
                        
                        pts = np.array(pts, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(page, [pts], True, gray_bgr, line_thickness)
                temp_token_index += 1
            if temp_token_index >= total_tokens:
                break
        
        # (No grid lines between tokens when using padding)
        
        # Second pass: Place tokens on top of the page at padded positions
        for row in range(tokens_per_col):
            for col in range(tokens_per_row):
                if token_index >= total_tokens:
                    break
                token_info = tokens_to_place[token_index]
                filename = token_info['filename']
                subfolder = token_info['subfolder']
                # Per-token padding override
                token_padding = padding
                if 'padding' in token_info:
                    token_padding = token_info['padding']
                token_padding_px = int(token_padding * ppi)
                # Calculate slot position (with centering offset)
                slot_x = x_margin_px + x_offset + col * (token_size_px + 2 * padding_px)
                slot_y = y_margin_px + y_offset + row * (token_size_px + 2 * padding_px)
                # Place token image centered in slot
                if subfolder:
                    token_path = img_folder / subfolder / filename
                else:
                    token_path = img_folder / filename
                token_img = cv2.imread(str(token_path), cv2.IMREAD_UNCHANGED)
                if token_img is None:
                    print(f"Warning: Could not load {filename}")
                    token_index += 1
                    continue
                
                # Apply brightness adjustments
                token_img = apply_brightness_adjustment(token_img, settings, token_info)
                
                # Store original size for scale calculation
                original_h, original_w = token_img.shape[:2]
                
                # Crop to the bounding box of non-transparent pixels
                if token_img.shape[2] == 4:
                    alpha_mask = token_img[:, :, 3] > 0
                    rows = np.any(alpha_mask, axis=1)
                    cols = np.any(alpha_mask, axis=0)
                    if rows.any() and cols.any():
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        token_img = token_img[y_min:y_max+1, x_min:x_max+1]
                
                # Get cropped size for scale calculation
                cropped_h, cropped_w = token_img.shape[:2]
                
                # Resize token to target size
                token_resized = cv2.resize(token_img, (token_size_px, token_size_px), interpolation=cv2.INTER_LANCZOS4)
                
                # Calculate scale ratio (from cropped to output)
                scale_ratio = token_size_px / max(cropped_w, cropped_h)
                # Compute placement
                x_pos = slot_x + token_padding_px
                y_pos = slot_y + token_padding_px
                # Place with alpha blending
                if token_resized.shape[2] == 4:
                    alpha = token_resized[:, :, 3] / 255.0
                    for c in range(3):
                        page[y_pos:y_pos+token_size_px, x_pos:x_pos+token_size_px, c] = (
                            alpha * token_resized[:, :, c] +
                            (1 - alpha) * page[y_pos:y_pos+token_size_px, x_pos:x_pos+token_size_px, c]
                        )
                    page[y_pos:y_pos+token_size_px, x_pos:x_pos+token_size_px, 3] = 255
                else:
                    page[y_pos:y_pos+token_size_px, x_pos:x_pos+token_size_px, :3] = token_resized
                
                # Draw brightness adjustment text below the token in the padding area
                brightness_text = token_info.get('brightness_adjustment', '')
                if token_padding_px > 5:  # Only if we have padding space
                    # Build info text with brightness and scale
                    info_parts = []
                    if brightness_text:
                        info_parts.append(f"BA:{brightness_text}")
                    info_parts.append(f"S:{scale_ratio:.2f}x")
                    info_parts.append(f"PPI:{ppi}")
                    
                    info_text = " | ".join(info_parts)
                    
                    # Calculate text position (bottom of padding area)
                    text_x = slot_x + token_padding_px
                    text_y = slot_y + token_size_px + token_padding_px + token_padding_px - 2  # Bottom of padding
                    
                    # Use smallest font size, scale based on PPI
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = max(0.2, token_padding_px / 80.0)  # Scale with padding
                    font_thickness = 1
                    text_color = (100, 100, 100, 255)  # Gray
                    
                    # Get text size to center it
                    (text_width, text_height), baseline = cv2.getTextSize(
                        info_text, font, font_scale, font_thickness
                    )
                    
                    # Center text horizontally in the token area
                    text_x_centered = slot_x + token_padding_px + (token_size_px - text_width) // 2
                    
                    # Ensure text fits in padding area
                    if text_height + 2 <= token_padding_px:
                        cv2.putText(page, info_text, 
                                  (text_x_centered, text_y), 
                                  font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                
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
    templates_dir = script_dir / "img" / "templates"
    metadata_path = script_dir / "img" / "image_metadata.json"
    
    if not metadata_path.exists():
        print(f"Error: Metadata not found at {metadata_path}")
        print("Please run process_tokens.py first to generate the metadata.")
    elif not templates_dir.exists():
        print(f"Error: Templates folder not found at {templates_dir}")
        print("Please create img/templates/ and add template folders with print_format_template.json files.")
    else:
        # Find all template folders
        template_folders = [f for f in templates_dir.iterdir() if f.is_dir()]
        
        if not template_folders:
            print(f"Error: No template folders found in {templates_dir}")
            print("Please create folders with print_format_template.json files.")
        else:
            print(f"Found {len(template_folders)} template folder(s)")
            print("=" * 50)
            
            # Process each template folder
            for template_folder in template_folders:
                template_path = template_folder / "print_format_template.json"
                
                if not template_path.exists():
                    print(f"\nSkipping {template_folder.name}: No print_format_template.json found")
                    continue
                
                print(f"\n{'#'*50}")
                print(f"# START: Processing template '{template_folder.name}'")
                print(f"{'#'*50}")
                
                # Delete all existing PNG files in template folder before generating new ones
                for png_file in template_folder.glob('*.png'):
                    png_file.unlink()
                    print(f"Deleted: {png_file.name}")
                
                # Update template with calculations
                template = update_template_calculations(template_path)
                
                settings = template['settings']
                width = settings['print_width']
                height = settings['print_height']
                token_size = settings['token_size']
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
                    
                    # Create output filename with dimensions, token size, and PPI (replace decimal points)
                    width_str = str(width).replace('.', '_')
                    height_str = str(height).replace('.', '_')
                    token_size_str = str(token_size).replace('.', '_')
                    output_filename = f"output_{width_str}x{height_str}_{token_size_str}in_{ppi}ppi.png"
                    output_path = template_folder / output_filename
                    
                    # Generate the layout
                    create_print_layout(template, output_path, metadata_path)
                
                print(f"\n{'#'*50}")
                print(f"# COMPLETE: Template '{template_folder.name}'")
                print(f"{'#'*50}")

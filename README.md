# Token Image Background Remover & Print Layout Generator

A Python toolkit to automatically process circular token images (like game tokens, coins, badges) by removing backgrounds and generating printable layouts.

## Overview

This tool provides two main functions:

1. **Background Removal** - Detects circular tokens using edge detection, removes backgrounds, and makes them transparent
2. **Print Layout Generation** - Arranges processed tokens into printable pages with customizable settings

## Features

- Automatic circular token detection using edge detection
- Background removal with transparent output
- Handles splotchy or inconsistent backgrounds
- Generates detailed metadata (position, dimensions, circularity, aspect ratio, border color)
- Creates customizable print templates
- Generates multi-page print layouts with proper spacing and margins
- High-quality image scaling (LANCZOS4 interpolation)
- Automatic cropping to token content (removes padding)
- Embeds PPI metadata in output images

## Setup

### Prerequisites

- Python 3.7 or higher

### Installation

1. Install the required dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

### Step 1: Process Token Images

Place your token images in the `img/input/` folder and run:

```powershell
python process_tokens.py
```

**Supported formats:** PNG, JPG, JPEG, BMP, TIFF

**Output:**
- Processed images in `img/output/` (PNG with transparency)
- `img/image_metadata.json` - Detailed analysis of each token
- `img/print_format_template.json` - Print configuration template

### Step 2: Configure Print Settings (Optional)

Edit `img/print_format_template.json` to customize:

**Settings:**
- `print_width` / `print_height` - Page dimensions (default: 8.5 x 11 inches)
- `x_margin` / `y_margin` - Page margins (default: 0.5 inches)
- `x_spacer` / `y_spacer` - Space between tokens (default: 0.2 inches)
- `token_size` - Token print size (default: 1.0 inch)
- `ppi` - Print resolution (default: 300)

**Token Quantities:**
- Edit the `quantity` field for each token in `tokens_quantity_list`

### Step 3: Generate Print Layout

Run the layout generator:

```powershell
python generate_print_layout.py
```

**Output:**
- Single page: `img/output_8_5x11_0_300ppi.png`
- Multiple pages: `img/output_8_5x11_0_300ppi_page1.png`, `page2.png`, etc.

The script automatically:
- Calculates how many tokens fit per page
- Updates template with calculated values
- Generates all needed pages
- Embeds PPI metadata for proper printing

## How It Works

### Background Removal (process_tokens.py)

1. **Edge Detection**: Uses Canny edge detection to find strong edges
2. **Contour Analysis**: Identifies the largest contour (the token)
3. **Circle Detection**: Creates a circular mask around the token
4. **Mask Erosion**: Removes background edge pixels (5 iterations)
5. **Mask Smoothing**: Applies morphological operations and Gaussian blur
6. **Transparency**: Sets alpha channel based on the mask

### Metadata Collection

For each token, the script analyzes:
- **Dimensions**: Image width/height in pixels
- **Position**: X/Y center coordinates
- **Diameter**: Detected circle diameter
- **Circularity**: Metric from 0-1 (1.0 = perfect circle)
- **Aspect Ratio**: Major/minor axis ratio (detects stretching)
- **Ellipse Metrics**: Major axis, minor axis, angle
- **Border Color**: Average RGB color of outermost pixels

### Print Layout Generation (generate_print_layout.py)

1. **Calculate Grid**: Determines tokens per row/column based on settings
2. **Content Cropping**: Crops each token to remove transparent padding
3. **High-Quality Scaling**: Resizes tokens using LANCZOS4 interpolation
4. **Grid Placement**: Arranges tokens with proper margins and spacing
5. **Alpha Blending**: Properly blends transparent tokens onto white background
6. **Multi-Page Support**: Generates multiple pages when needed
7. **PPI Metadata**: Embeds resolution information in PNG files

## Project Structure

```
manage_token_images/
├── img/
│   ├── input/                 # Place input images here
│   ├── output/                # Processed images with transparency
│   ├── image_metadata.json    # Token analysis data
│   ├── print_format_template.json  # Print configuration
│   └── output_*.png           # Generated print layouts
├── process_tokens.py          # Background removal script
├── generate_print_layout.py   # Print layout generator
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Print Quality

**PPI Guidelines:**
- **72-96 PPI**: Screen viewing, web use
- **150 PPI**: Minimum acceptable for printing
- **300 PPI**: Professional quality (recommended, default)
- **600 PPI**: Very high quality, laser printers

**Source Image Quality:**
Your 1024x1024 source images provide excellent quality for:
- 1 inch tokens at 300 PPI (uses 300x300 pixels)
- 3.4 inch tokens at 300 PPI (uses full resolution)

## Technical Details

### Circularity Calculation
Uses the formula: `4π × area / perimeter²`
- 1.0 = Perfect circle
- 0.9-1.0 = Very circular
- 0.8-0.9 = Mostly circular
- <0.8 = Increasingly irregular

### Aspect Ratio Detection
Fits an ellipse to detect stretching:
- 1.0 = No stretching (circular)
- \>1.0 = Stretched/elliptical
- `ellipse_angle` indicates direction (0°=horizontal, 90°=vertical)

## Notes

- The script assumes tokens have strong circular borders
- Background colors can vary and be splotchy - the algorithm focuses on edges
- All output images are PNG format to support transparency
- Original images are not modified; processed versions are saved separately
- Tokens are automatically cropped to content for optimal layout spacing

## Troubleshooting

**Background not removed properly:**
- Ensure the token has a clear, strong border
- Check that the token is the largest circular object in the image
- Verify image quality is sufficient for edge detection
- Adjust erosion iterations in `process_tokens.py` if needed

**Tokens have spacing when spacers are 0:**
- This is now fixed - tokens are cropped to content before placement
- Transparent padding is automatically removed

**Print layout doesn't match expected size:**
- Check PPI metadata is embedded (script does this automatically)
- Verify print settings in `print_format_template.json`
- Some applications may ignore or override PPI metadata

# Token Image Background Remover & Print Layout Generator

A Python toolkit to automatically process circular token images (like game tokens, coins, badges) by removing backgrounds and generating printable layouts with advanced brightness adjustment for photo paper printing.

## Overview

This tool provides two main functions:

1. **Background Removal** - Detects circular tokens using edge detection, removes backgrounds, and makes them transparent
2. **Print Layout Generation** - Arranges processed tokens into printable pages with customizable settings and brightness adjustments

## Features

- Automatic circular token detection using edge detection
- Background removal with transparent output
- Handles splotchy or inconsistent backgrounds
- Generates detailed metadata (position, dimensions, circularity, aspect ratio, border color, shape classification)
- **Advanced brightness adjustment** for photo paper printing (gamma, shadow lift, midtone boost, highlight boost)
- **Compact brightness notation** with array expansion support
- Creates customizable print templates with multiple template folders
- Generates multi-page print layouts with proper spacing and configurable padding
- High-quality image scaling (LANCZOS4 interpolation)
- Automatic cropping to token content (removes padding)
- Shape-aware border rendering (circles for circular tokens, rounded rectangles for squares)
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

### Step 2: Configure Print Settings

Templates are stored in `img/templates/<template_name>/print_format_template.json`. You can have multiple templates for different layouts.

Edit template files to customize:

**Settings:**
- `print_width` / `print_height` - Page dimensions (e.g., 8.5 x 11 inches, 6 x 4 inches)
- `x_margin` / `y_margin` - Page margins (in inches)
- `padding` - Space between tokens (in inches, applied outward from token edges)
- `token_size` - Token print size (in inches)
- `ppi` - Print resolution (can be single value or array like `[300, 600]`)
- `brightness_adjustment` - Default brightness settings (optional, see Brightness Adjustment section)

**Token Quantities:**
- Edit the `quantity` field for each token in `tokens_quantity_list`
- Add `subfolder` field to reference tokens in subdirectories
- Override `padding` per-token if needed
- Override `brightness_adjustment` per-token for fine control

### Brightness Adjustment for Photo Paper Printing

Photo paper often prints darker than expected. The brightness adjustment system allows you to compensate by brightening midtones, highlights, and optionally lifting shadows while preserving deep blacks.

**Brightness Format:** `"gamma-shadow_lift-midtone_boost-highlight_boost"`

**Parameters:**
- `gamma` (0.5-3.0): Overall midtone brightness. Values >1.0 brighten, typical range 1.2-2.2
- `shadow_lift` (0.0-0.2): Lifts dark areas. Use 0.0 to keep blacks black, 0.05-0.1 for slight lift
- `midtone_boost` (0.8-2.0): Multiplier for browns/grays. 1.0 = no change, >1.0 = brighter
- `highlight_boost` (0.8-2.5): Multiplier for light areas toward white. 1.0 = no change

**Examples:**
```json
"brightness_adjustment": "1-0-1-1"        // No adjustment (baseline)
"brightness_adjustment": "1.5-0-1-1"      // Mild gamma boost
"brightness_adjustment": "1.8-0-1.3-1.5"  // Strong gamma + boost mids/highlights
"brightness_adjustment": "2-0.1-1.5-2"    // Extreme + slight shadow lift
```

**Array Expansion:**
Save lines by using array notation. Arrays expand via cartesian product:
```json
// Instead of listing 4 separate strings:
"brightness_adjustment": ["1-0-1-1", "1.5-0-1-1", "1.8-0-1-1", "2-0-1-1"]

// Use compact array notation (expands to 4 strings):
"brightness_adjustment": ["[1,1.5,1.8,2]-0-1-1"]

// Multiple arrays expand multiplicatively (expands to 8 strings):
"brightness_adjustment": ["[1.5,1.8]-0-[1.3,1.5]-[1.5,1.8]"]
// Generates: 1.5-0-1.3-1.5, 1.5-0-1.3-1.8, 1.5-0-1.5-1.5, 1.5-0-1.5-1.8,
//            1.8-0-1.3-1.5, 1.8-0-1.3-1.8, 1.8-0-1.5-1.5, 1.8-0-1.5-1.8

// Mix expanded and regular strings in same array:
"brightness_adjustment": [
  "[1,1.5,2]-0-1-1",      // Expands to 3
  "1.8-0-1.5-1.8",        // Single string
  "2.2-0-1.6-2"           // Single string
]
// Total: 5 tokens with different brightness settings
```

**Template-level vs Token-level:**
- Set `brightness_adjustment` in `settings` for default applied to all tokens
- Override per-token in `tokens_quantity_list` for specific adjustments
- Use arrays to generate test variations for finding optimal settings

### Step 3: Generate Print Layout

Run the layout generator:

```powershell
python generate_print_layout.py
```

The script processes all templates in `img/templates/` and generates output in each template folder.

**Output:**
- Template folder: `img/templates/<template_name>/`
- Single page: `output_<width>x<height>_<ppi>ppi.png`
- Multiple pages: `output_<width>x<height>_<ppi>ppi_page1.png`, `page2.png`, etc.
- Multiple PPI values: Generates separate output files for each PPI

The script automatically:
- Calculates how many tokens fit per page based on size, padding, and margins
- Updates template with calculated values
- Generates all needed pages
- Embeds PPI metadata for proper printing
- Applies brightness adjustments per token or template defaults

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
- **Shape Classification**: Categorizes as circle, rounded_square, square, rounded_rect, or rect
- **Circularity**: Metric from 0-1 (1.0 = perfect circle, ≥0.89 = circle, ≥0.80 = rounded_square)
- **Aspect Ratio**: Major/minor axis ratio (1.0 = circular, >1.0 = elliptical)
- **Ellipse Metrics**: Major axis, minor axis, angle
- **Border Color**: Average RGB color of outermost pixels (1px and 3px depth)
- **Subfolder**: Source subfolder for organizing tokens by type

### Print Layout Generation (generate_print_layout.py)

1. **Calculate Grid**: Determines tokens per row/column based on settings and padding
2. **Shape-Aware Borders**: Draws circles for circular tokens, rounded rectangles for square tokens
3. **Brightness Adjustment**: Applies gamma, shadow lift, midtone boost, and highlight boost
4. **Content Cropping**: Crops each token to remove transparent padding
5. **High-Quality Scaling**: Resizes tokens using LANCZOS4 interpolation
6. **Grid Placement**: Arranges tokens with proper margins and padding
7. **Alpha Blending**: Properly blends transparent tokens onto white background
8. **Multi-Page Support**: Generates multiple pages when needed
9. **PPI Metadata**: Embeds resolution information in PNG files

### Brightness Adjustment Algorithm

The brightness adjustment system uses a multi-stage approach:

1. **Gamma Correction**: Applies power-law transformation `output = input^(1/gamma)`, which primarily affects midtones
2. **Tonal Masking**: Creates separate masks for shadows (0-0.33), midtones (0.33-0.67), and highlights (0.67-1.0)
3. **Shadow Lift**: Adds constant value to dark areas to lift blacks
4. **Midtone Boost**: Multiplies midtone values to brighten browns/grays
5. **Highlight Boost**: Multiplies highlight values to push light grays toward white
6. **Clipping**: Ensures all values remain in valid 0-1 range

This preserves deep blacks while selectively brightening problem areas that print too dark on photo paper.

## Project Structure

```
manage_token_images/
├── img/
│   ├── input/                      # Place input images here (organized in subfolders)
│   │   ├── subfolder1/
│   │   └── subfolder2/
│   ├── output/                     # Processed images with transparency
│   │   ├── subfolder1/
│   │   └── subfolder2/
│   ├── templates/                  # Print template configurations
│   │   ├── 6x4/
│   │   │   ├── print_format_template.json
│   │   │   └── output_*.png       # Generated layouts
│   │   └── 8_5x11/
│   │       ├── print_format_template.json
│   │       └── output_*.png
│   ├── image_metadata.json         # Token analysis data
│   └── print_format_template.json  # Root template (generated by process_tokens.py)
├── process_tokens.py               # Background removal script
├── generate_print_layout.py        # Print layout generator
├── requirements.txt                # Python dependencies
└── README.md                       # This file
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

**Circular tokens getting rounded square borders:**
- Ensure `process_tokens.py` was run to regenerate metadata with shape classification
- Check that circularity threshold (≥0.89) is appropriate for your tokens
- Verify metadata uses composite key `(subfolder, filename)` to handle duplicate names

**Print layout doesn't match expected size:**
- Check PPI metadata is embedded (script does this automatically)
- Verify print settings in template JSON files
- Some applications may ignore or override PPI metadata

**Brightness adjustment not working:**
- Verify brightness string format: `"gamma-shadow_lift-midtone_boost-highlight_boost"`
- Check that values are numeric (use `1.5` not `1,5`)
- Array expansion requires proper syntax: `"[1,1.5,2]-0-1-1"`
- Ensure token-level settings override template-level as expected

**Token count mismatch:**
- Remember that array notation expands: `"[1,2]-0-1-[1,2]"` creates 4 tokens
- When using `brightness_adjustment` array, `quantity` is ignored
- Check calculated values in template after running script

## Tips & Best Practices

- **Photo Paper Printing**: Start with `"1.5-0-1.3-1.5"` and adjust based on test prints
- **Test Variations**: Use array notation to generate multiple brightness settings in one print
- **Organize by Type**: Use subfolders in `img/input/` to keep token types organized
- **Multiple Templates**: Create separate templates for different page sizes or layouts
- **High PPI**: Use 600 PPI for professional quality, especially for small tokens
- **Shape Classification**: Circular tokens (circularity ≥0.89) get circle borders, others get rounded rectangles

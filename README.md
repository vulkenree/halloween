# Stick Figure Video Processor

A Python-based video processing system that detects people in video frames using MediaPipe pose detection and converts them into black and white stick figure animations.

## Features

- Real-time pose detection using MediaPipe
- Black and white stick figure rendering from detected poses
- Optional 90-degree clockwise rotation (enabled by default)
- Optional motion blur with glow effect for enhanced visual appeal
- Frame sampling for faster processing (process fewer frames to speed up)
- Performance metrics and progress reporting

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for package management.

### Prerequisites

- Python 3.8 or higher
- `uv` package manager

### Installation

1. Install `uv` if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install project dependencies:
   ```bash
   uv sync
   ```

3. Activate the virtual environment (if needed):
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

## Usage

### Preparing Test Videos

Place your test video files in the `test_videos/` directory (or any location of your choice). Supported formats include `.mp4`, `.mov`, `.avi`, `.mkv`, and other formats supported by OpenCV.

### Basic Processing

Process a video to create a stick figure animation:

```bash
uv run python main.py --input test_videos/your_video.mp4 --output test_videos/output.mp4
```

Example with a `.MOV` file:

```bash
uv run python main.py -i test_videos/emma_walking.MOV -o test_videos/output.mp4
```

Or with a full path:

```bash
uv run python main.py -i /path/to/input.mp4 -o /path/to/output.mp4
```

### Command-line Options

- `--input` / `-i`: Path to input video file (required)
- `--output` / `-o`: Path to output video file (required)
- `--no-rotate`: Disable rotation (rotation is enabled by default - rotates output 90° clockwise)
- `--motion-blur`: Enable motion blur with glow effect for stick figures
- `--frame-sampling`: Frame sampling rate (0.0-1.0, default: 1.0)
  - `1.0` = Process all frames (no downsampling)
  - `0.5` = Process 50% of frames (every 2nd frame) - ~2x faster
  - `0.25` = Process 25% of frames (every 4th frame) - ~4x faster

### Usage Examples

**Process with all default settings (rotation enabled, all frames):**
```bash
uv run python main.py -i input.mp4 -o output.mp4
```

**Process without rotation:**
```bash
uv run python main.py -i input.mp4 -o output.mp4 --no-rotate
```

**Process with motion blur effect:**
```bash
uv run python main.py -i input.mp4 -o output.mp4 --motion-blur
```

**Process faster by sampling 50% of frames:**
```bash
uv run python main.py -i input.mp4 -o output.mp4 --frame-sampling 0.5
```

**Combine multiple options:**
```bash
uv run python main.py -i input.mp4 -o output.mp4 --motion-blur --frame-sampling 0.25 --no-rotate
```

## Project Structure

- `main.py` - Main entry point and command-line interface
- `ghost_processor.py` - Core processing module with pose detection and stick figure rendering
- `pyproject.toml` - Project configuration and dependencies
- `test_videos/` - Directory for test video files (you can place your videos here)

## Output

The processed video will contain:
- White stick figures on a black background
- Same resolution as input (or rotated if rotation is enabled)
- Adjusted frame rate if frame sampling is used (maintains correct timing)
- Motion trails and glow effects if motion blur is enabled

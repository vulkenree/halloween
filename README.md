# Halloween Ghost Video Processor

A Python-based video processing system that detects people in video frames, extracts them, and creates ghost effect versions with lagged/shadow projections for Halloween displays.

## Features

- Real-time person detection and segmentation using MediaPipe
- Configurable ghost effects (transparency, color tinting, outline)
- Lagged shadow effects with configurable frame delays
- Process pre-recorded videos or live camera feeds (future)

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

### Processing a Video

Process a pre-recorded video:

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
- `--transparency` / `-t`: Ghost transparency level (0.0-1.0, default: 0.5)
- `--lag-frames` / `-l`: Number of frames delay for shadow effect (default: 30)
- `--ghost-color`: RGB color for ghost effect (default: white/transparent)

## Project Structure

- `main.py` - Main entry point and command-line interface
- `ghost_processor.py` - Core processing module with person detection and ghost effects
- `pyproject.toml` - Project configuration and dependencies
- `test_videos/` - Directory for test video files (you can place your videos here)

## Future Phases

- Phase 2: Live camera integration
- Phase 3: Real-time projection output
- Phase 4: Advanced effects and optimizations


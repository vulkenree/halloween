"""
Main entry point for stick figure video processor.
"""

import argparse
import sys
from ghost_processor import process_video


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert input video to black and white stick figure video"
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to output video file'
    )
    
    args = parser.parse_args()
    
    try:
        process_video(
            input_path=args.input,
            output_path=args.output
        )
    except Exception as e:
        print(f"Error processing video: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

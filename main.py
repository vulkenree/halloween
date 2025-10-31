"""
Main entry point for Halloween Ghost Video Processor.
"""

import argparse
import sys
from ghost_processor import process_ghost_video, process_witch_video


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create ghost or witch effects from input video"
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
    
    parser.add_argument(
        '--witch',
        action='store_true',
        help='Create witch mode (witch head + accessories) instead of ghost mode'
    )
    
    args = parser.parse_args()
    
    try:
        if args.witch:
            # Witch mode
            process_witch_video(
                input_path=args.input,
                output_path=args.output
            )
        else:
            # Ghost mode (default)
            process_ghost_video(
                input_path=args.input,
                output_path=args.output
            )
    except Exception as e:
        print(f"Error processing video: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
Main entry point for Halloween Ghost Video Processor.
"""

import argparse
import sys
from ghost_processor import process_video, process_stick_figure_to_ghost


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect people in video and render stick figures, or create ghost effects from stick figure video"
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
        '--ghost',
        action='store_true',
        help='Process stick figure video to create ghost effects (requires stick figure video as input)'
    )
    
    parser.add_argument(
        '--glow-intensity',
        type=int,
        default=15,
        help='Intensity of glow effect for ghost (default: 15)'
    )
    
    parser.add_argument(
        '--ghost-color',
        type=str,
        default=None,
        help='Ghost color as "R,G,B" (default: light blue 200,200,255)'
    )
    
    parser.add_argument(
        '--fill-transparency',
        type=float,
        default=0.3,
        help='Transparency of ghost fill (0.0-1.0, default: 0.3)'
    )
    
    parser.add_argument(
        '--ultra-ghostly',
        action='store_true',
        help='Apply enhanced blur and ghostly effects (for ghost processing mode)'
    )
    
    parser.add_argument(
        '--spooky-head',
        action='store_true',
        help='Replace stick figure head with spooky ghost head (for stick figure mode)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.ghost:
            # Ghost processing mode
            ghost_color = None
            if args.ghost_color:
                parts = args.ghost_color.split(',')
                if len(parts) != 3:
                    raise ValueError("Ghost color must have 3 components: R,G,B")
                r, g, b = [int(x.strip()) for x in parts]
                ghost_color = (b, g, r)  # Convert RGB to BGR
            
            if args.fill_transparency < 0.0 or args.fill_transparency > 1.0:
                raise ValueError("Fill transparency must be between 0.0 and 1.0")
            
            process_stick_figure_to_ghost(
                input_path=args.input,
                output_path=args.output,
                glow_intensity=args.glow_intensity,
                ghost_color=ghost_color,
                fill_transparency=args.fill_transparency,
                ultra_ghostly=args.ultra_ghostly
            )
        else:
            # Normal stick figure processing
            process_video(
                input_path=args.input,
                output_path=args.output,
                spooky_head=args.spooky_head
            )
    except Exception as e:
        print(f"Error processing video: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

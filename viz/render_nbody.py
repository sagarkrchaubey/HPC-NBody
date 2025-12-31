#run command
#python3 render_nbody_pro.py --input "$CSV" --output "temp_frames" --limit $LIMIT --fps 30 --dpi 150 --video "$VIDEO_NAME" \
    # --keep-frames


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import multiprocessing
import numpy as np
import argparse
import subprocess
import shutil

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Parallel N-Body Renderer with Auto-Video")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    parser.add_argument("-o", "--output", default="render_output", help="Output directory for frames")
    parser.add_argument("--fps", type=int, default=30, help="Video framerate")
    parser.add_argument("--dpi", type=int, default=100, help="Image quality (DPI)")
    parser.add_argument("--limit", type=float, default=2.0, help="Axis limits (-limit to +limit)")
    parser.add_argument("--video", default="simulation.mp4", help="Output video filename")
    parser.add_argument("--keep-frames", action="store_true", help="Keep PNG frames after video generation")
    return parser.parse_args()

def render_frame(args):
    """
    Renders a single frame. 
    """
    step, df_step, output_dir, axis_limit, dpi = args
    
    # Setup Plot
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Dark background
    plt.style.use('dark_background')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black') 
    
    # --- Advanced Visualization: Color by Velocity ---
    # Calculate speed v = sqrt(vx^2 + vy^2 + vz^2)
    # If velocity data is missing, fallback to Cyan
    if {'vx', 'vy', 'vz'}.issubset(df_step.columns):
        v = np.sqrt(df_step['vx']**2 + df_step['vy']**2 + df_step['vz']**2)
        # Normalize speed for color map (0 to 2.0 roughly)
        colors = v
        cmap = 'plasma' # 'plasma' goes from blue/purple to orange/yellow (very sci-fi)
    else:
        colors = 'cyan'
        cmap = None

    # Plot Particles
    sc = ax.scatter(df_step['x'], df_step['y'], df_step['z'], 
                    s=5, c=colors, cmap=cmap, alpha=0.8, edgecolors='none')

    # Fix Camera and Axis
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)
    
    ax.set_title(f"Step {int(step)}", color='white', fontsize=8)
    
    # Hide Axes for clean look
    ax.axis('off')
    
    # Save file
    filename = os.path.join(output_dir, f"frame_{int(step):05d}.png")
    plt.savefig(filename, bbox_inches='tight', facecolor='black', pad_inches=0)
    plt.close(fig)
    
    return filename

def main():
    args = parse_arguments()
    
    print(f"--- N-Body Renderer ---")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}/")
    print(f"Limit:  {args.limit}")
    
    # 1. Read Data
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
        return

    # Check columns
    if not {'x', 'y', 'z', 'step'}.issubset(df.columns):
        print("Error: CSV missing required columns (step, x, y, z)")
        return

    # 2. Prepare Directory
    if os.path.exists(args.output):
        print(f"Warning: Cleaning existing directory '{args.output}'...")
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    # 3. Group Tasks
    grouped = df.groupby('step')
    tasks = []
    for step, group in grouped:
        tasks.append((step, group, args.output, args.limit, args.dpi))
    
    total_frames = len(tasks)
    num_cores = multiprocessing.cpu_count()
    print(f"Rendering {total_frames} frames using {num_cores} cores...")

    # 4. Parallel Render
    with multiprocessing.Pool(processes=num_cores) as pool:
        for i, _ in enumerate(pool.imap_unordered(render_frame, tasks), 1):
            sys.stdout.write(f"\rProgress: {i}/{total_frames}")
            sys.stdout.flush()
    print("\nRendering Images Complete.")

    # 5. Video Generation (FFmpeg)
    ffmpeg_cmd = [
        "ffmpeg", "-y",                 # Overwrite output
        "-framerate", str(args.fps),    # FPS
        "-pattern_type", "glob",        # Use glob for filenames
        "-i", f"{args.output}/*.png",   # Input pattern
        "-c:v", "libx264",              # Codec
        "-pix_fmt", "yuv420p",          # Pixel format for compatibility
        "-crf", "18",                   # High quality
        args.video                      # Output filename
    ]

    print(f"Generating Video: {args.video}...")
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"SUCCESS: Video saved as '{args.video}'")
    except FileNotFoundError:
        print("Error: 'ffmpeg' not found. Please install ffmpeg or load the module.")
        print("Images are saved, but video creation failed.")
    except subprocess.CalledProcessError:
        print("Error: FFmpeg failed to generate video.")

    # 6. Cleanup (Optional)
    if not args.keep_frames:
        print("Cleaning up frames...")
        shutil.rmtree(args.output)
        print("Cleanup Done.")

if __name__ == "__main__":
    main()

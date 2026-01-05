import matplotlib
matplotlib.use('Agg') # Headless backend for Clusters/HPC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os
import subprocess
from multiprocessing import Pool, cpu_count
from scipy.spatial import cKDTree 

# ---------- CONFIGURATION ----------
CSV_FILE = "nbody_output_openmp_N2000.csv"
OUT_DIR = "frames_final_v2"
VIDEO_NAME = "galaxy_simulation_enhanced.mp4"

STEP_STRIDE = 1 
FPS = 30

# UPDATED: 'plasma' is much better for black backgrounds (starts purple/blue, not black)
COLOR_MAP = 'plasma' 

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- HELPER: DETECT COLUMNS ----------
def get_columns(df):
    """Normalize column names to handle variations like vx vs v_x"""
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    
    # Position
    mapping['x'] = cols.get('x', cols.get('pos_x', None))
    mapping['y'] = cols.get('y', cols.get('pos_y', None))
    mapping['z'] = cols.get('z', cols.get('pos_z', None))
    
    # Velocity (for Speed Graph)
    mapping['vx'] = cols.get('vx', cols.get('v_x', cols.get('vel_x', None)))
    mapping['vy'] = cols.get('vy', cols.get('v_y', cols.get('vel_y', None)))
    mapping['vz'] = cols.get('vz', cols.get('v_z', cols.get('vel_z', None)))
    
    return mapping

# ---------- FRAME RENDER FUNCTION ----------
def render_frame(args):
    step_val, df_step, frame_id, global_max_speed = args
    
    # 1. Data Extraction
    col_map = get_columns(df_step)
    
    # Positions
    pos = df_step[[col_map['x'], col_map['y'], col_map['z']]].values
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
    
    # Velocities (Calculate Speed Magnitude)
    if col_map['vx']:
        vels = df_step[[col_map['vx'], col_map['vy'], col_map['vz']]].values
        speeds = np.linalg.norm(vels, axis=1)
    else:
        # Fallback if no velocity data exists
        speeds = np.zeros(len(pos))

    # 2. Density Calculation (for glowing effect)
    tree = cKDTree(pos)
    dists, _ = tree.query(pos, k=min(8, len(pos))) 
    density = 1.0 / (dists[:, -1] + 1e-5)
    
    # Center of Mass (for tracking)
    com = np.mean(pos, axis=0)
    
    # Normalize density for color (log scale helps with contrast)
    v_min, v_max = np.percentile(density, 5), np.percentile(density, 99.5)

    # 3. Setup Figure
    fig = plt.figure(figsize=(20, 10), facecolor='black') 
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 1.2, 0.8])
    
    # --- VIEW 1: GLOBAL STRUCTURE (Rotating) ---
    ax1 = fig.add_subplot(gs[:, 0], projection='3d')
    ax1.set_facecolor('black')
    
    # Draw Particles (Increased size slightly for visibility)
    # "Bloom" layer (large, faint)
    ax1.scatter(xs, ys, zs, c=density, cmap=COLOR_MAP, vmin=v_min, vmax=v_max, 
                s=25, alpha=0.08, edgecolors='none')
    # Core layer (small, bright)
    ax1.scatter(xs, ys, zs, c=density, cmap=COLOR_MAP, vmin=v_min, vmax=v_max, 
                s=2.5, alpha=0.9, edgecolors='none')
    
    # Dynamic Limits
    limit = np.percentile(np.abs(pos), 99) * 1.5
    ax1.set_xlim(-limit, limit); ax1.set_ylim(-limit, limit); ax1.set_zlim(-limit, limit)
    ax1.set_axis_off()
    ax1.view_init(elev=20 * np.cos(frame_id * 0.01), azim=frame_id * 0.8)
    ax1.text2D(0.05, 0.95, "GLOBAL VIEW", transform=ax1.transAxes, color='white', fontsize=12, fontweight='bold')

    # --- VIEW 2: CORE ZOOM (With Magnification Bar) ---
    ax2 = fig.add_subplot(gs[:, 1], projection='3d')
    ax2.set_facecolor('black')
    
    ax2.scatter(xs, ys, zs, c=density, cmap=COLOR_MAP, vmin=v_min, vmax=v_max, 
                s=50, alpha=0.1, edgecolors='none')
    ax2.scatter(xs, ys, zs, c=density, cmap=COLOR_MAP, vmin=v_min, vmax=v_max, 
                s=8, alpha=1.0, edgecolors='none')
    
    # Zoom Logic
    z_limit = limit * 0.3 # 30% of global view
    ax2.set_xlim(com[0]-z_limit, com[0]+z_limit)
    ax2.set_ylim(com[1]-z_limit, com[1]+z_limit)
    ax2.set_zlim(com[2]-z_limit, com[2]+z_limit)
    ax2.set_axis_off()
    ax2.view_init(elev=10, azim=-frame_id * 1.2)
    
    # ADDED: Magnification Indicator
    mag_factor = limit / z_limit
    ax2.text2D(0.05, 0.95, f"TRACKING CORE", transform=ax2.transAxes, color='white', fontsize=12, fontweight='bold')
    # Draw a "Bar" using text (simplest way in 3D plots)
    ax2.text2D(0.05, 0.90, f"MAGNIFICATION: {mag_factor:.1f}x", 
               transform=ax2.transAxes, color='cyan', fontsize=14, fontweight='bold')
    # Visual bar
    ax2.text2D(0.05, 0.88, "▓▓▓▓▓▓▓▓", transform=ax2.transAxes, color='cyan', fontsize=10, alpha=0.7)

    # --- VIEW 3: DENSITY HEATMAP (Top Right) ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('black')
    # Changed cmap to 'inferno' for high contrast heatmap
    hb = ax3.hexbin(xs, ys, gridsize=50, cmap='inferno', bins='log', mincnt=1)
    ax3.set_aspect('equal')
    ax3.set_axis_off()
    ax3.set_title("XY PROJECTION DENSITY", color='white', fontsize=10)

    # --- VIEW 4: VELOCITY DISTRIBUTION (Bottom Right - "Speed Up Graph") ---
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor('black')
    
    if len(speeds) > 0:
        # Plot histogram of particle speeds
        n, bins, patches = ax4.hist(speeds, bins=50, color='orange', alpha=0.7, density=True)
        
        # Style
        ax4.set_title("PARTICLE VELOCITY DISTRIBUTION", color='white', fontsize=10)
        ax4.set_xlabel("Speed magnitude (|v|)", color='gray', fontsize=8)
        ax4.tick_params(axis='both', colors='gray', labelsize=8)
        for spine in ax4.spines.values(): spine.set_color('#444444')
        ax4.grid(axis='y', color='#333333', linestyle='--')
        
        # Keep x-axis static so the graph doesn't jump around, giving a sense of acceleration
        if global_max_speed > 0:
            ax4.set_xlim(0, global_max_speed * 0.8) 
    else:
        ax4.text(0.5, 0.5, "NO VELOCITY DATA", color='red', ha='center')

    # Global Title
    plt.suptitle(f"N-BODY SIMULATION | STEP {step_val} | N=2000", color="white", fontsize=20, y=0.96)
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95)
    
    fname = f"{OUT_DIR}/frame_{frame_id:05d}.png"
    plt.savefig(fname, facecolor='black', dpi=90) # slightly higher DPI
    plt.close(fig)
    return fname

# ---------- VIDEO ENCODING ----------
def make_video():
    print(f"\n[FFmpeg] Encoding {VIDEO_NAME}...")
    cmd = ['ffmpeg', '-y', '-framerate', str(FPS), '-i', f'{OUT_DIR}/frame_%05d.png',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'medium', VIDEO_NAME]
    try:
        subprocess.run(cmd, check=True)
        print(f"Video saved as {VIDEO_NAME}")
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg or check your path.")

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found!")
        exit()

    print(f"Reading data and starting render on {cpu_count()} cores...")
    
    # Load Data
    df = pd.read_csv(CSV_FILE)
    
    # Normalize step column name
    cols = {c.lower(): c for c in df.columns}
    step_col = cols.get('step', None)
    if not step_col:
        print("Error: Could not find a 'step' column in CSV.")
        exit()

    unique_steps = sorted(df[step_col].unique())[::STEP_STRIDE]
    
    # Pre-calculate global max speed for consistent graph axis
    col_map = get_columns(df)
    global_max_speed = 0
    if col_map['vx']:
        # Estimate max speed from a sample to set graph limits
        sample_step = unique_steps[len(unique_steps)//2]
        sample_df = df[df[step_col] == sample_step]
        vx = sample_df[col_map['vx']].values
        vy = sample_df[col_map['vy']].values
        vz = sample_df[col_map['vz']].values
        global_max_speed = np.max(np.linalg.norm(np.column_stack((vx, vy, vz)), axis=1))
        print(f"Calculated max particle speed approx: {global_max_speed:.2f}")

    # Prepare Tasks
    render_tasks = []
    for i, s_val in enumerate(unique_steps):
        step_data = df[df[step_col] == s_val]
        render_tasks.append((s_val, step_data, i, global_max_speed))

    # Multiprocessing Render
    with Pool(cpu_count()) as p:
        p.map(render_frame, render_tasks)
    
    print(f"Successfully rendered {len(render_tasks)} frames.")
    make_video()
    print("Done!")

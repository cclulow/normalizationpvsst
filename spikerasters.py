import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

data_path = '/media/maclean/Storage/42R/072424/runs/suite2p/plane0'

# === Load Suite2p outputs ===
F = np.load(os.path.join(data_path, 'F.npy'))          # Raw fluorescence
Fneu = np.load(os.path.join(data_path, 'Fneu.npy'))    # Neuropil
iscell = np.load(os.path.join(data_path, 'iscell.npy'))[:, 0].astype(bool)  # Cell mask

# Optional spike inference
try:
    spks = np.load(os.path.join(data_path, 'spks.npy'))  # Inferred spikes
    has_spikes = True
except FileNotFoundError:
    print("No spks.npy found. Skipping spike analysis.")
    spks = None
    has_spikes = False

alpha = 1.0
Fcorr = F - alpha * Fneu
Fcorr_cells = Fcorr[iscell]
if has_spikes:
    spks_cells = spks[iscell]

frame_start = 0
frame_end = Fcorr_cells.shape[1] // 4

#  can update this later
selected_cells = [17, 18,23,36]
# === Normalize traces for plotting ===
def normalize_trace(trace):
    return (trace - np.min(trace)) / (np.max(trace) - np.min(trace))

# === Plot Panel b: ΔF/F traces ===
def plot_traces(cells=selected_cells, start=0, end=20000):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Shade stimulus periods
    stim_duration = 30
    stim_interval = 60
    stim_times = [(s, s + stim_duration) for s in range(start, end, stim_interval)]
    
    for s, e in stim_times:
        ax.axvspan(s - start, e - start, color='lightblue', alpha=0.3)
    
    # Plot traces
    for i, cell in enumerate(cells):
        trace = normalize_trace(Fcorr_cells[cell][start:end])
        ax.plot(trace + i * 2, color='black', linewidth=1)
    
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('ΔF/F (offset)')
    ax.set_title(f'ΔF/F Traces ({start}–{end} frames)')
    plt.tight_layout()
    plt.show()

# === Plot Panel c: spike raster ===
def plot_raster(cells=selected_cells, start=0, end=500):
    if not has_spikes:
        print("Spikes not available.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Shade stimulus periods
    stim_duration = 30
    stim_interval = 60
    total_frames = spks_cells.shape[1]
    stim_times = [(start, start + stim_duration) for start in range(0, total_frames, stim_interval)]
    
    for start, end in stim_times:
        ax.axvspan(start, end, color='lightblue', alpha=0.3)
    
    # Plot raster
    for i, cell in enumerate(cells):
        spike_times = np.where(spks_cells[cell] > 0)[0]
        ax.vlines(spike_times, i + 0.5, i + 1.5, color='black', linewidth=0.6)
    
    ax.set_yticks(np.arange(1, len(cells)+1))
    ax.set_yticklabels([f'Cell {i}' for i in cells])
    ax.set_xlabel('Time (frames)')
    ax.set_title('Raster Plot with Stim Periods (Panel c)')
    plt.tight_layout()
    plt.show()

def plot_trace_with_spikes(cells=selected_cells, start=0, end=5000):
    if not has_spikes:
        print("Spikes not available.")
        return

    fig, ax = plt.subplots(figsize=(10, 8)) 
    fontname = 'Nimbus Sans'  
    plt.rcParams.update({'font.size': 18})

    label_fontsize = 24
    tick_fontsize = 24
    title_fontsize = 30

    trace_height = 2.0
    spike_height = 1.0
    spacing = 0.5

    for i, cell in enumerate(cells):
        base_y = i * (trace_height + spike_height + spacing)

        trace = normalize_trace(Fcorr_cells[cell][start:end])
        ax.plot(trace * trace_height + base_y + spike_height, color='black', linewidth=1)

        spike_times = np.where(spks_cells[cell][start:end] > 12)[0]
        ax.vlines(spike_times, base_y, base_y + spike_height, color='red', linewidth=1.5)

    frame_ticks = ax.get_xticks()
    second_ticks = frame_ticks / 30  # 30 Hz = 30 frames per second
    ax.set_xticks(frame_ticks)
    ax.set_xticklabels([f"{t:.1f}" for t in second_ticks], fontsize=tick_fontsize, fontname=fontname)
    ax.set_xlabel('Time (s)', fontsize=label_fontsize, fontname=fontname)
    ax.set_xlim(-200, 6000)
    ax.set_ylabel('Neuron', fontsize=label_fontsize, fontname=fontname)
    ax.set_title('ΔF/F Traces with Spikes Below', fontsize=title_fontsize, fontname=fontname)

    ax.set_yticks([i * (trace_height + spike_height + spacing) + trace_height / 2 for i in range(len(cells))])
    ax.set_yticklabels([f'Cell {i}' for i in cells], fontsize=tick_fontsize, fontname=fontname)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    for label in ax.get_xticklabels():
        label.set_fontname(fontname)

    plt.tight_layout()
    plt.show()
    

# === Call plotting functions ===
#plot_traces(start=0, end=20000)
#plot_raster(start=0, end=500)
plot_trace_with_spikes(start=0, end=6000)
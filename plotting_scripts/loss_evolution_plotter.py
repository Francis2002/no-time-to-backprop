import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
import re
import argparse

def find_matching_files(online_path, tbptt_path, file_pattern, filter=None):
    """Find matching configuration files across FBPTT and TBPTT results."""
    online_files = glob.glob(str(online_path / file_pattern))
    tbptt_files = glob.glob(str(tbptt_path / file_pattern))
    
    matched_files = []
    for on_file in online_files:
        # Extract config from filename
        config_name, _ = parse_config_from_path(on_file)

        if filter and filter not in config_name:
            continue
        fb_config = Path(on_file).name
        tb_file = str(tbptt_path / Path(on_file).relative_to(online_path))
        
        if os.path.exists(tb_file):
            matched_files.append((on_file, tb_file, fb_config))
    
    return matched_files

def parse_config_from_path(file_path):
    """Parse configuration parameters from file path."""
    path_parts = Path(file_path).parts
    
    # Extract memory, hidden, layers, activation from directory name
    dir_name = [p for p in path_parts if p.startswith('memory_')][0]
    memory = re.search(r'memory_(\d+)', dir_name).group(1)
    hidden = re.search(r'hidden_(\d+)', dir_name).group(1)
    layers = re.search(r'layers_(\d+)', dir_name).group(1)
    act = re.search(r'act_(\w+)', dir_name).group(1)
    
    # Extract flags from filename
    filename = Path(file_path).name
    prenorm = 'True' if 'prenorm_True' in filename else 'False'
    encoder = 'True' if 'encoder_True' in filename else 'False'
    layerout = 'True' if 'layerout_True' in filename else 'False'
    extraskip = 'True' if 'extraskip_True' in filename else 'False'
    decoder = 'MLP' if 'decoder_MLP' in filename else 'NONE'
    mixing = 'full' if 'mixing_full' in filename else 'none'
    nonlinrec = 'True' if 'nonlinrec_True' in filename else 'False'

    return f"M{memory}_H{hidden}_L{layers}_Act{act}_PN{prenorm}_Enc{encoder}_LO{layerout}_ES{extraskip}_Dec{decoder}_Mix{mixing}_NLR{nonlinrec}", memory

def plot_loss_comparison(online_file, tbptt_file, output_dir, skip_epochs=50):
    """Plot loss comparison between ONLINE and TBPTT methods."""
    try:
        online_loss = np.load(online_file)
        tbptt_loss = np.load(tbptt_file)
        
        # Skip initial epochs
        online_loss = online_loss[skip_epochs:]
        tbptt_loss = tbptt_loss[skip_epochs:]

        epochs = np.arange(skip_epochs, len(online_loss) + skip_epochs)

        # Create plot with logarithmic scale
        plt.figure(figsize=(10, 6))
        plt.semilogy(epochs, online_loss, label='ONLINE', linewidth=2)
        plt.semilogy(epochs, tbptt_loss, label='TBPTT', linewidth=2, linestyle='--')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss (log scale)')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        # Extract configuration for title
        config_name, memory = parse_config_from_path(online_file)
        plt.title(f'Loss Comparison - {config_name}')
        plt.legend()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(output_dir, f'memory_{memory}')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'loss_comparison_{config_name}.png')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Created loss comparison plot: {output_file}")
        return True
    except Exception as e:
        print(f"Error creating plot for {online_file}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Plot loss comparison between FBPTT and TBPTT')
    parser.add_argument('--output_dir', type=str, default='plots/loss_comparisons', 
                        help='Output directory for plots')
    parser.add_argument('--filter', type=str, default='', 
                        help='Filter for specific configurations (e.g., "M1_H32" or "NLRTrue")')
    parser.add_argument('--skip', type=int, default=50, 
                        help='Number of initial epochs to skip')
    args = parser.parse_args()
    
    # Base paths for results
    online_base = Path('results/bitcoin_alpha_ONLINE')
    tbptt_base = Path('results/bitcoin_alpha_TBPTT')

    # Find matching loss trajectory files
    file_pattern = "**/loss_trajectory/*.npy"
    matching_files = find_matching_files(online_base, tbptt_base, file_pattern, args.filter)
    
    print(f"Found {len(matching_files)} matching configuration pairs")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots for each configuration
    success_count = 0
    for fbptt_file, tbptt_file, config in matching_files:
        if plot_loss_comparison(fbptt_file, tbptt_file, args.output_dir, args.skip):
            success_count += 1
    
    print(f"Successfully created {success_count} loss comparison plots in {args.output_dir}")

if __name__ == "__main__":
    main()
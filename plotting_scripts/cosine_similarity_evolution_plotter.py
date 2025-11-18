import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
import re
from scipy.spatial.distance import cosine
import argparse

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    if np.all(v1 == 0) or np.all(v2 == 0):
        return 0  # Return 0 if either vector is all zeros
    return 1 - cosine(v1, v2)  # scipy.spatial.distance.cosine returns 1-cosine_similarity

def flatten_array(arr):
    """Flatten an array or list of arrays."""
    if isinstance(arr, list):
        # Handle list of arrays
        if all(isinstance(x, (list, np.ndarray)) for x in arr):
            return np.concatenate([flatten_array(x) for x in arr])
        else:
            return np.array(arr).flatten()
    elif isinstance(arr, np.ndarray):
        return arr.flatten()
    else:
        return np.array([arr]).flatten()

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
    
    return f"M{memory}_H{hidden}_L{layers}_Act{act}_PN{prenorm}_Enc{encoder}"

def extract_grad_data(grads_file, num_layers):
    """Extract gradient data from a file."""
    try:
        all_grads = np.load(grads_file, allow_pickle=True)
        
        # Initialize arrays to store gradients for each parameter type
        nu_grads = []
        theta_grads = []
        gamma_grads = []
        B_re_grads = []
        B_im_grads = []
        
        # Extract gradients for each epoch
        for epoch_grads in all_grads:
            nu_epoch = []
            theta_epoch = []
            gamma_epoch = []
            B_re_epoch = []
            B_im_epoch = []
            
            # Extract gradients for each layer
            for layer_idx in range(num_layers):
                layer_key = f'layer_{layer_idx}'
                if layer_key in epoch_grads:
                    layer_grads = epoch_grads[layer_key]
                    
                    if 'nu_grads' in layer_grads and layer_grads['nu_grads'] is not None:
                        nu_epoch.append(flatten_array(layer_grads['nu_grads']))
                    
                    if 'theta_grads' in layer_grads and layer_grads['theta_grads'] is not None:
                        theta_epoch.append(flatten_array(layer_grads['theta_grads']))
                    
                    if 'gamma_log_grads' in layer_grads and layer_grads['gamma_log_grads'] is not None:
                        gamma_epoch.append(flatten_array(layer_grads['gamma_log_grads']))
                    
                    if 'B_re_grads' in layer_grads and layer_grads['B_re_grads'] is not None:
                        B_re_epoch.append(flatten_array(layer_grads['B_re_grads']))
                    
                    if 'B_im_grads' in layer_grads and layer_grads['B_im_grads'] is not None:
                        B_im_epoch.append(flatten_array(layer_grads['B_im_grads']))
            
            # Concatenate all layer gradients for this epoch
            if nu_epoch:
                nu_grads.append(np.concatenate(nu_epoch))
            if theta_epoch:
                theta_grads.append(np.concatenate(theta_epoch))
            if gamma_epoch:
                gamma_grads.append(np.concatenate(gamma_epoch))
            if B_re_epoch:
                B_re_grads.append(np.concatenate(B_re_epoch))
            if B_im_epoch:
                B_im_grads.append(np.concatenate(B_im_epoch))
        
        return {
            'nu': nu_grads,
            'theta': theta_grads,
            'gamma': gamma_grads,
            'B_re': B_re_grads,
            'B_im': B_im_grads
        }
    except Exception as e:
        print(f"Error extracting gradients from {grads_file}: {str(e)}")
        return None

def calculate_cosine_similarity(grads1, grads2):
    """Calculate cosine similarity between two sets of gradients over epochs."""
    if not grads1 or not grads2:
        return None
    
    min_epochs = min(len(grads1), len(grads2))
    similarities = []
    
    for i in range(min_epochs):
        if len(grads1[i]) > 0 and len(grads2[i]) > 0:
            sim = cosine_similarity(grads1[i], grads2[i])
            similarities.append(sim)
        else:
            similarities.append(0)
    
    return similarities

def plot_cosine_similarity(fbptt_file, tbptt_file, output_dir, num_layers, skip_epochs=50):
    """Plot cosine similarity between FBPTT and TBPTT gradients."""
    try:
        config_name = parse_config_from_path(fbptt_file)
        
        # Extract gradients
        fbptt_grads = extract_grad_data(fbptt_file, num_layers)
        tbptt_grads = extract_grad_data(tbptt_file, num_layers)
        
        if not fbptt_grads or not tbptt_grads:
            print(f"Missing gradient data for {config_name}")
            return False
        
        # Calculate cosine similarities for each parameter type
        similarities = {}
        for param_type in ['nu', 'theta', 'gamma', 'B_re', 'B_im']:
            if param_type in fbptt_grads and param_type in tbptt_grads:
                similarities[param_type] = calculate_cosine_similarity(
                    fbptt_grads[param_type], 
                    tbptt_grads[param_type]
                )
        
        # Skip initial epochs
        for param_type in similarities:
            if similarities[param_type] and len(similarities[param_type]) > skip_epochs:
                similarities[param_type] = similarities[param_type][skip_epochs:]
        
        # Plot similarities
        plt.figure(figsize=(12, 8))
        epochs = np.arange(skip_epochs, skip_epochs + len(next(iter(similarities.values()))))
        
        for param_type, sim_values in similarities.items():
            if sim_values:
                plt.plot(epochs, sim_values, label=f'{param_type}', linewidth=2)
        
        plt.xlabel('Epochs')
        plt.ylabel('Cosine Similarity')
        plt.title(f'Gradient Cosine Similarity - {config_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'cosine_similarity_{config_name}.png')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Created cosine similarity plot: {output_file}")
        return True
    except Exception as e:
        print(f"Error creating cosine similarity plot for {config_name}: {str(e)}")
        return False

def find_matching_files(fbptt_path, tbptt_path, file_pattern):
    """Find matching configuration files across FBPTT and TBPTT results."""
    fbptt_files = glob.glob(str(fbptt_path / file_pattern))
    tbptt_files = glob.glob(str(tbptt_path / file_pattern))
    
    matched_files = []
    for fb_file in fbptt_files:
        # Extract config from filename
        fb_config = Path(fb_file).name
        tb_file = str(tbptt_path / Path(fb_file).relative_to(fbptt_path))
        
        if os.path.exists(tb_file):
            matched_files.append((fb_file, tb_file, fb_config))
    
    return matched_files

def main():
    parser = argparse.ArgumentParser(description='Plot gradient cosine similarity between FBPTT and TBPTT')
    parser.add_argument('--output_dir', type=str, default='plots/cosine_similarity', 
                        help='Output directory for plots')
    parser.add_argument('--skip', type=int, default=50, 
                        help='Number of initial epochs to skip')
    args = parser.parse_args()
    
    # Base paths for results
    fbptt_base = Path('results/toy_task_FBPTT')
    tbptt_base = Path('results/toy_task_TBPTT')
    
    # Find matching gradient files
    file_pattern = "**/all_grads/*.npy"
    matching_files = find_matching_files(fbptt_base, tbptt_base, file_pattern)
    
    print(f"Found {len(matching_files)} matching gradient file pairs")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract number of layers from file paths
    success_count = 0
    for fbptt_file, tbptt_file, config in matching_files:
        path = Path(fbptt_file)
        directory_name = path.parent.parent.name
        num_layers = int(re.search(r'layers_(\d+)', directory_name).group(1))
        
        if plot_cosine_similarity(fbptt_file, tbptt_file, args.output_dir, num_layers, args.skip):
            success_count += 1
    
    print(f"Successfully created {success_count} cosine similarity plots in {args.output_dir}")

if __name__ == "__main__":
    main()
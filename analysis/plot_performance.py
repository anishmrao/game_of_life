import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_performance(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path, header=None, 
                    names=['approach', 'num_cells', 'mean_time', 'iterations'])
    
    # Get unique approaches for consistent colors
    approaches = df['approach'].unique()
    
    # Function to save a plot
    def save_plot(fig, filename):
        output_path = os.path.join(os.path.dirname(csv_path), filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    # 1. Log Scale Plot
    plt.figure(figsize=(12, 8))
    for name in approaches:
        group = df[df['approach'] == name]
        plt.plot(group['num_cells'], group['mean_time'], 'o-', label=name)
    
    plt.title('Performance Comparison (Log Scale)')
    plt.xlabel('Number of Cells')
    plt.ylabel('Mean Time per Iteration (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(title='Approach')
    plt.tight_layout()
    save_plot(plt.gcf(), 'performance_log_scale.png')
    
    # 2. Linear Scale Plot
    plt.figure(figsize=(12, 8))
    for name in approaches:
        group = df[df['approach'] == name]
        plt.plot(group['num_cells'], group['mean_time'], 'o-', label=name)
    
    plt.title('Performance Comparison (Linear Scale)')
    plt.xlabel('Number of Cells')
    plt.ylabel('Mean Time per Iteration (seconds)')
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend(title='Approach')
    plt.tight_layout()
    save_plot(plt.gcf(), 'performance_linear_scale.png')
    
    # 3. Numba vs CUDA Comparison
    if 'numba' in approaches and 'cuda' in approaches:
        plt.figure(figsize=(12, 8))
        for name in ['numba', 'cuda']:
            group = df[df['approach'] == name]
            plt.plot(group['num_cells'], group['mean_time'], 'o-', label=name)
        
        plt.title('Numba vs CUDA Performance Comparison')
        plt.xlabel('Number of Cells')
        plt.ylabel('Mean Time per Iteration (seconds)')
        plt.grid(True, ls="--", alpha=0.3)
        plt.legend(title='Approach')
        plt.tight_layout()
        save_plot(plt.gcf(), 'performance_numba_vs_cuda.png')
    else:
        print("Warning: Could not create Numba vs CUDA comparison - one or both approaches not found in data")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python plot_performance.py <path_to_csv>")
        sys.exit(1)
    
    plot_performance(sys.argv[1])

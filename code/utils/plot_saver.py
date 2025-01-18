import os

def save_plot(fig, filename, results_plots_dir=os.path.join('..', '..', 'results', 'plots')):
    """
    Save the given Matplotlib figure to the specified directory.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to save.
        filename (str): The name of the file (e.g., 'my_plot.png').
        results_plots_dir (str): Directory to save the plot.
                                 Defaults to '../../results/plots'.
    """
    # Ensure the directory exists
    os.makedirs(results_plots_dir, exist_ok=True)

    # Create the full file path
    file_path = os.path.join(results_plots_dir, filename)

    # Save the figure
    fig.savefig(file_path, bbox_inches='tight')
    print(f"Plot saved to: {file_path}")
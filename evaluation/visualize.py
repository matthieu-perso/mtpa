import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(file_path: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_accuracy_bar_plot(results: dict, output_path: str):
    """Create a bar plot of model accuracies."""
    # Set style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Prepare data
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    # Create figure and axis
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    bars = plt.bar(models, accuracies)
    
    # Customize plot
    plt.title('Model Accuracy Comparison', fontsize=14, pad=20)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Load results
    results_path = Path('WorldValuesBench/benchmark_results.json')
    results = load_results(results_path)
    
    # Create output directory if it doesn't exist
    output_dir = Path('WorldValuesBench/visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    create_accuracy_bar_plot(results, output_dir / 'model_accuracies.png')
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main() 
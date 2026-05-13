import re
import matplotlib.pyplot as plt
import os

def parse_log(file_path, results):
    with open(file_path, 'r') as f:
        for line in f:
            # Pattern to extract all metrics
            # Task: X, Round: Y | TrainLoss: L | EvalLoss: E | Acc: A% | Macro-F1: MA% | Weighted-F1: W% | Micro-F1: MI%
            regex = r"Round:\s*(\d+)\s*\|\s*TrainLoss:\s*([\d.]+)\s*\|\s*EvalLoss:\s*([\d.]+)\s*\|\s*Acc:\s*([\d.]+)%\s*\|\s*Macro-F1:\s*([\d.]+)%\s*\|\s*Weighted-F1:\s*([\d.]+)%\s*\|\s*Micro-F1:\s*([\d.]+)%"
            match = re.search(regex, line)
            if match:
                round_idx = int(match.group(1))
                results[round_idx] = {
                    'train_loss': float(match.group(2)),
                    'eval_loss': float(match.group(3)),
                    'acc': float(match.group(4)),
                    'macro_f1': float(match.group(5)),
                    'weighted_f1': float(match.group(6)),
                    'micro_f1': float(match.group(7))
                }

def main():
    log_files = [
        r"c:\Users\LENOVO\Desktop\glfc\sim-glfc-gpu\ketquatrain180round\log_1_round0-79.txt",
        r"c:\Users\LENOVO\Desktop\glfc\sim-glfc-gpu\ketquatrain180round\log_1_round80-142.txt",
        r"c:\Users\LENOVO\Desktop\glfc\sim-glfc-gpu\ketquatrain180round\log_1_round142-179.txt"
    ]
    
    results = {}
    for f in log_files:
        if os.path.exists(f):
            parse_log(f, results)

    if not results:
        print("No data found!")
        return

    sorted_rounds = sorted(results.keys())
    rounds = sorted_rounds
    
    accs = [results[r]['acc'] for r in rounds]
    macro_f1s = [results[r]['macro_f1'] for r in rounds]
    weighted_f1s = [results[r]['weighted_f1'] for r in rounds]
    micro_f1s = [results[r]['micro_f1'] for r in rounds]
    
    train_losses = [results[r]['train_loss'] for r in rounds]
    eval_losses = [results[r]['eval_loss'] for r in rounds]

    plt.style.use('seaborn-v0_8-muted')
    fig, ax1 = plt.subplots(figsize=(15, 8), dpi=300)
    
    # Colors for performance metrics (Left Y-Axis)
    color_acc = '#1f77b4'       # Blue
    color_macro = '#2ca02c'     # Green
    color_weighted = '#9467bd'  # Purple
    color_micro = '#17becf'     # Cyan (Note: Micro-F1 is usually same as Acc)
    
    # Colors for Losses (Right Y-Axis)
    color_train = '#ff7f0e'     # Orange
    color_eval = '#d62728'      # Red
    color_grid = '#e0e0e0'

    # Plot Accuracy and F1 Scores
    ax1.plot(rounds, accs, color=color_acc, label='Accuracy (%)', linewidth=3, alpha=0.9)
    ax1.plot(rounds, macro_f1s, color=color_macro, label='Macro-F1 (%)', linewidth=1.5, linestyle='--', alpha=0.8)
    ax1.plot(rounds, weighted_f1s, color=color_weighted, label='Weighted-F1 (%)', linewidth=1.5, linestyle=':', alpha=0.8)
    # Micro-F1 is often identical to Accuracy, so we plot it as a very thin line or skip if identical
    if any(m != a for m, a in zip(micro_f1s, accs)):
        ax1.plot(rounds, micro_f1s, color=color_micro, label='Micro-F1 (%)', linewidth=1, linestyle='-.', alpha=0.7)

    ax1.set_xlabel('Round (30 rounds per Task)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 105)
    
    task_boundaries = [i * 30 for i in range(7)]
    ax1.set_xticks(task_boundaries)
    ax1.grid(True, linestyle='--', alpha=0.6, color=color_grid)

    # Plot Losses on second axis
    ax2 = ax1.twinx()
    ax2.plot(rounds, train_losses, color=color_train, label='Train Loss', linestyle='--', linewidth=1.2, alpha=0.6)
    ax2.plot(rounds, eval_losses, color=color_eval, label='Eval Loss', linestyle='-.', linewidth=1.2, alpha=0.6)
    ax2.set_ylabel('Loss Value', fontsize=12, fontweight='bold')

    # Task highlighting
    task_size = 30
    num_tasks = 6
    for i in range(num_tasks):
        start = i * task_size
        end = (i + 1) * task_size
        if i > 0:
            ax1.axvline(x=start, color='#7f8c8d', linestyle='-', linewidth=1.2, alpha=0.7)
        if i % 2 == 0:
            ax1.axvspan(start, end, color='#f7f9f9', alpha=0.5, zorder=0)
        label_x = start + task_size / 2
        ax1.text(label_x, 101, f'TASK {i}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold', color='#34495e',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    plt.title('Federated Learning Metrics: Accuracy, F1 Scores & Loss', fontsize=16, fontweight='bold', pad=20)
    
    # Combined Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=True)

    plt.tight_layout()
    
    output_path = r"c:\Users\LENOVO\Desktop\glfc\sim-glfc-gpu\ketquatrain180round\training_plot_full_metrics.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Full metrics plot saved to {output_path}")

if __name__ == "__main__":
    main()

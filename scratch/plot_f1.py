import re
import matplotlib.pyplot as plt
import os

def parse_log(file_path, results):
    with open(file_path, 'r') as f:
        for line in f:
            regex = r"Round:\s*(\d+)\s*\|\s*TrainLoss:\s*([\d.]+)\s*\|\s*EvalLoss:\s*([\d.]+)\s*\|\s*Acc:\s*([\d.]+)%\s*\|\s*Macro-F1:\s*([\d.]+)%\s*\|\s*Weighted-F1:\s*([\d.]+)%\s*\|\s*Micro-F1:\s*([\d.]+)%"
            match = re.search(regex, line)
            if match:
                round_idx = int(match.group(1))
                results[round_idx] = {
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
    
    macro_f1s = [results[r]['macro_f1'] for r in rounds]
    weighted_f1s = [results[r]['weighted_f1'] for r in rounds]
    micro_f1s = [results[r]['micro_f1'] for r in rounds]
    accs = [results[r]['acc'] for r in rounds]

    plt.style.use('seaborn-v0_8-muted')
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    
    # High-contrast colors for F1 scores
    color_macro = '#d62728'     # Red
    color_weighted = '#9467bd'  # Purple
    color_micro = '#1f77b4'     # Blue
    color_acc = '#2ca02c'       # Green (for reference)
    color_grid = '#e0e0e0'

    # Plot F1 Scores with solid lines for better visibility
    ax.plot(rounds, macro_f1s, color=color_macro, label='Macro-F1 (%)', linewidth=2, marker='o', markersize=3, markevery=5)
    ax.plot(rounds, weighted_f1s, color=color_weighted, label='Weighted-F1 (%)', linewidth=2, marker='s', markersize=3, markevery=5)
    ax.plot(rounds, micro_f1s, color=color_micro, label='Micro-F1 (%)', linewidth=2, marker='^', markersize=3, markevery=5)
    
    # Accuracy as a background reference
    ax.plot(rounds, accs, color='#7f8c8d', label='Accuracy (%) [Ref]', linewidth=1, alpha=0.3)

    ax.set_xlabel('Round (30 rounds per Task)', fontsize=12, fontweight='bold')
    ax1_ylabel = ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    
    task_boundaries = [i * 30 for i in range(7)]
    ax.set_xticks(task_boundaries)
    ax.grid(True, linestyle='--', alpha=0.6, color=color_grid)

    # Task highlighting
    task_size = 30
    num_tasks = 6
    for i in range(num_tasks):
        start = i * task_size
        end = (i + 1) * task_size
        if i > 0:
            ax.axvline(x=start, color='#7f8c8d', linestyle='-', linewidth=1.5, alpha=0.7)
        if i % 2 == 0:
            ax.axvspan(start, end, color='#f7f9f9', alpha=0.5, zorder=0)
        
        label_x = start + task_size / 2
        ax.text(label_x, 101, f'TASK {i}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color='#2c3e50')

    plt.title('Detailed F1 Scores Analysis (Macro, Micro, Weighted)', fontsize=16, fontweight='bold', pad=25)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#bdc3c7', fontsize=11)

    plt.tight_layout()
    
    output_path = r"c:\Users\LENOVO\Desktop\glfc\sim-glfc-gpu\ketquatrain180round\training_plot_f1_only.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"F1-only plot saved to {output_path}")

if __name__ == "__main__":
    main()

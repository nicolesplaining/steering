"""
Visualize ICL+COT experiment results for GSM8K or MATH.
"""
import json
import matplotlib.pyplot as plt
import sys
import glob

# Find the most recent results file, or use argument
if len(sys.argv) > 1:
    results_file = sys.argv[1]
else:
    # Find most recent results file
    files = glob.glob("icl_results*.json")
    if not files:
        print("No results files found!")
        sys.exit(1)
    results_file = max(files)  # Most recent by name (timestamp)
    print(f"Using: {results_file}")

# Load results
with open(results_file, "r") as f:
    data = json.load(f)

# Extract data
dataset = data.get("dataset", "GSM8K")
k_values = data["k_values"]
thinking = [data["thinking_results"][str(k)]["accuracy"] * 100 for k in k_values]
non_thinking = [data["non_thinking_results"][str(k)]["accuracy"] * 100 for k in k_values]

# Create figure with dark theme for MATH (competition level)
if dataset == "MATH":
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    text_color = '#eee'
    think_color = '#ff6b6b'
    no_think_color = '#4ecdc4'
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    text_color = 'black'
    think_color = '#6366f1'
    no_think_color = '#10b981'

# Plot lines
ax.plot(k_values, thinking, 'o-', linewidth=2.5, markersize=10, 
        color=think_color, label='Thinking Mode', zorder=3)
ax.plot(k_values, non_thinking, 's-', linewidth=2.5, markersize=10, 
        color=no_think_color, label='Non-Thinking Mode', zorder=3)

# Add value labels
for i, (k, t, nt) in enumerate(zip(k_values, thinking, non_thinking)):
    ax.annotate(f'{t:.1f}%', (k, t), textcoords="offset points", 
                xytext=(0, 12), ha='center', fontsize=9, color=think_color, fontweight='bold')
    ax.annotate(f'{nt:.1f}%', (k, nt), textcoords="offset points", 
                xytext=(0, -18), ha='center', fontsize=9, color=no_think_color, fontweight='bold')

# Highlight best points
best_think_idx = thinking.index(max(thinking))
best_no_think_idx = non_thinking.index(max(non_thinking))
ax.scatter([k_values[best_think_idx]], [thinking[best_think_idx]], 
           s=200, color=think_color, alpha=0.3, zorder=2)
ax.scatter([k_values[best_no_think_idx]], [non_thinking[best_no_think_idx]], 
           s=200, color=no_think_color, alpha=0.3, zorder=2)

# Styling
ax.set_xlabel('Number of ICL Examples (k)', fontsize=12, fontweight='bold', color=text_color)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color=text_color)

# Calculate improvements
think_improvement = max(thinking) - thinking[0]
no_think_improvement = max(non_thinking) - non_thinking[0]

if dataset == "MATH":
    ax.set_title(f'ICL+COT on MATH-500 (Competition Level)\nQwen3-8B · n=200 problems', 
                 fontsize=14, fontweight='bold', pad=15, color=text_color)
    ax.set_ylim(35, 85)
    # Key finding for MATH
    textstr = (f'⚠️ Non-thinking mode dominates ({non_thinking[0]:.1f}% vs {thinking[0]:.1f}%)\n'
               f'📈 Thinking improves: {thinking[0]:.1f}% → {max(thinking):.1f}% (+{think_improvement:.1f}%)\n'
               f'📈 Non-thinking best at k={k_values[best_no_think_idx]}: {max(non_thinking):.1f}%')
    props = dict(boxstyle='round,pad=0.5', facecolor='#2d3748', edgecolor='#4ecdc4', alpha=0.95)
else:
    ax.set_title('ICL+COT Improves Math Reasoning on GSM8K\nQwen3-8B · n=200 problems', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(82, 100)
    textstr = f'✓ ICL+COT boosts accuracy up to +{max(think_improvement, no_think_improvement):.1f}%\n✓ Best k={k_values[best_think_idx]} for thinking, k={k_values[best_no_think_idx]} for non-thinking'
    props = dict(boxstyle='round,pad=0.5', facecolor='#fef3c7', edgecolor='#f59e0b', alpha=0.9)

ax.set_xticks(k_values)
ax.set_xlim(-0.5, 9.5)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', color='gray' if dataset == "MATH" else None)
ax.set_axisbelow(True)

# Tick colors for dark theme
if dataset == "MATH":
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color('#444')

# Legend
legend = ax.legend(loc='lower right' if dataset != "MATH" else 'upper left', 
                   fontsize=11, framealpha=0.95)
if dataset == "MATH":
    legend.get_frame().set_facecolor('#2d3748')
    for text in legend.get_texts():
        text.set_color(text_color)

# Add annotation box with key finding
ax.text(0.02, 0.98 if dataset != "MATH" else 0.02, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top' if dataset != "MATH" else 'bottom', bbox=props, color=text_color)

# Tight layout and save
plt.tight_layout()
output_base = f'icl_results_{dataset.lower()}_plot'
plt.savefig(f'{output_base}.png', dpi=150, bbox_inches='tight', 
            facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig(f'{output_base}.pdf', bbox_inches='tight', 
            facecolor=fig.get_facecolor(), edgecolor='none')
print(f"Saved: {output_base}.png and {output_base}.pdf")

# Print summary
print(f"\n{'='*50}")
print(f"RESULTS SUMMARY: {dataset}")
print(f"{'='*50}")
print(f"\nThinking Mode:")
print(f"  Baseline (k=0): {thinking[0]:.1f}%")
print(f"  Best (k={k_values[best_think_idx]}): {max(thinking):.1f}%")
print(f"  Improvement: +{think_improvement:.1f}%")
print(f"\nNon-Thinking Mode:")
print(f"  Baseline (k=0): {non_thinking[0]:.1f}%")
print(f"  Best (k={k_values[best_no_think_idx]}): {max(non_thinking):.1f}%")
print(f"  Improvement: +{no_think_improvement:.1f}%")
print(f"\n{'='*50}")
if dataset == "MATH":
    print("🔥 KEY FINDING: Non-thinking mode CRUSHES thinking mode!")
    print(f"   Gap at k=0: {non_thinking[0] - thinking[0]:.1f}% difference")
    print("   This suggests the model's internal reasoning may be")
    print("   interfering with learned solution patterns on MATH.")
print(f"{'='*50}")

plt.show()

import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Load predictions
examples = []
with open('./hypothesis_only/eval_predictions.jsonl') as f:
    for line in f:
        examples.append(json.loads(line))

label_names = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

# ============================================
# FIGURE 1: Confusion Matrix
# ============================================
confusion = defaultdict(lambda: defaultdict(int))
for ex in examples:
    true = ex['label']
    pred = ex['predicted_label']
    confusion[true][pred] += 1

# Create confusion matrix
matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        matrix[i][j] = confusion[i][j]

# Normalize by row (true labels)
matrix_norm = matrix / matrix.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(matrix_norm, cmap='Blues', vmin=0, vmax=1)

# Add text annotations
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, f'{matrix[i, j]:.0f}\n({matrix_norm[i, j]:.2%})',
                      ha="center", va="center", color="black" if matrix_norm[i, j] < 0.5 else "white",
                      fontsize=10)

ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(3))
ax.set_xticklabels([label_names[i] for i in range(3)])
ax.set_yticklabels([label_names[i] for i in range(3)])
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Hypothesis-Only Model Confusion Matrix', fontsize=14, fontweight='bold')

plt.colorbar(im, ax=ax, label='Proportion')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved confusion_matrix.png")
plt.close()

# ============================================
# FIGURE 2: Negation Analysis
# ============================================
negation_words = {'no', 'not', 'nobody', 'nothing', 'never', 'none', 'nowhere'}

def has_negation(text):
    return bool(negation_words & set(text.lower().split()))

negation_stats = defaultdict(lambda: {'with_neg': 0, 'without_neg': 0})
for ex in examples:
    pred = ex['predicted_label']
    if has_negation(ex['hypothesis']):
        negation_stats[pred]['with_neg'] += 1
    else:
        negation_stats[pred]['without_neg'] += 1

# Calculate percentages
labels = [label_names[i] for i in range(3)]
with_neg_pcts = []
for i in range(3):
    total = negation_stats[i]['with_neg'] + negation_stats[i]['without_neg']
    pct = negation_stats[i]['with_neg'] / total * 100
    with_neg_pcts.append(pct)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, with_neg_pcts, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
ax.set_ylabel('% of Predictions with Negation Words', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_title('Negation Words Strongly Correlate with Contradiction', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(with_neg_pcts) * 1.2)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, with_neg_pcts)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('negation_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved negation_analysis.png")
plt.close()

# ============================================
# FIGURE 3: Hypothesis Length Distribution
# ============================================
lengths_by_label = defaultdict(list)
for ex in examples:
    length = len(ex['hypothesis'].split())
    lengths_by_label[ex['predicted_label']].append(length)

fig, ax = plt.subplots(figsize=(10, 5))
positions = [0, 1, 2]
colors = ['#2ecc71', '#3498db', '#e74c3c']

# Create violin plots
parts = ax.violinplot([lengths_by_label[i] for i in range(3)],
                      positions=positions,
                      widths=0.7,
                      showmeans=True,
                      showmedians=True)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)

ax.set_xticks(positions)
ax.set_xticklabels([label_names[i] for i in range(3)])
ax.set_ylabel('Hypothesis Length (words)', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_title('Hypothesis Length Distribution by Predicted Label', fontsize=14, fontweight='bold')

# Add mean values as text
for i in range(3):
    mean_len = np.mean(lengths_by_label[i])
    ax.text(i, max([max(lengths_by_label[j]) for j in range(3)]) * 0.95,
            f'μ={mean_len:.1f}',
            ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('hypothesis_length.png', dpi=300, bbox_inches='tight')
print("✓ Saved hypothesis_length.png")
plt.close()

# ============================================
# FIGURE 4: Comparison - Hypothesis-Only vs Full Model
# ============================================
# This assumes you have both models' results

fig, ax = plt.subplots(figsize=(8, 5))

models = ['Random\nBaseline', 'Hypothesis\nOnly', 'Full Model\n(P+H)']
accuracies = [33.3, 89.0, 88.9]  # Update with your actual full model accuracy
colors = ['#95a5a6', '#e74c3c', '#3498db']

bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Dataset Artifacts: Premise Adds No Value', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.axhline(y=33.3, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random chance')

# Add value labels on bars
for bar, val in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved model_comparison.png")
plt.close()

# ============================================
# FIGURE 5: Per-Class Accuracy
# ============================================
accuracy_by_label = defaultdict(lambda: {'correct': 0, 'total': 0})
for ex in examples:
    true_label = ex['label']
    pred_label = ex['predicted_label']
    accuracy_by_label[true_label]['total'] += 1
    if true_label == pred_label:
        accuracy_by_label[true_label]['correct'] += 1

labels = [label_names[i] for i in range(3)]
accuracies = []
for i in range(3):
    acc = accuracy_by_label[i]['correct'] / accuracy_by_label[i]['total'] * 100
    accuracies.append(acc)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, accuracies, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('True Label', fontsize=12)
ax.set_title('Hypothesis-Only Model: Per-Class Accuracy', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.axhline(y=89.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Overall accuracy')

# Add value labels on bars
for bar, val in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.legend()
plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
print("✓ Saved per_class_accuracy.png")
plt.close()

print("\n✓✓✓ All visualizations complete! ✓✓✓")
print("\nGenerated files:")
print("  1. confusion_matrix.png")
print("  2. negation_analysis.png")
print("  3. hypothesis_length.png")
print("  4. model_comparison.png")
print("  5. per_class_accuracy.png")

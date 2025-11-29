import json
from collections import Counter, defaultdict

# Load predictions
examples = []
with open('./hypothesis_only/eval_predictions.jsonl') as f:
    for line in f:
        examples.append(json.loads(line))

# Analyze predictions by label
label_names = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

# 1. Check accuracy by true label
accuracy_by_label = defaultdict(lambda: {'correct': 0, 'total': 0})
for ex in examples:
    true_label = ex['label']
    pred_label = ex['predicted_label']
    accuracy_by_label[true_label]['total'] += 1
    if true_label == pred_label:
        accuracy_by_label[true_label]['correct'] += 1

print("=== Accuracy by True Label ===")
for label in [0, 1, 2]:
    acc = accuracy_by_label[label]['correct'] / accuracy_by_label[label]['total']
    print(f"{label_names[label]}: {acc:.3f} ({accuracy_by_label[label]['correct']}/{accuracy_by_label[label]['total']})")

# 2. Find common words in each predicted class
def get_words(text):
    return set(text.lower().split())

word_counts_by_pred = {0: Counter(), 1: Counter(), 2: Counter()}
for ex in examples:
    pred = ex['predicted_label']
    words = get_words(ex['hypothesis'])
    word_counts_by_pred[pred].update(words)

print("\n=== Top 20 Words in Each Predicted Class ===")
for label in [0, 1, 2]:
    print(f"\n{label_names[label].upper()}:")
    for word, count in word_counts_by_pred[label].most_common(20):
        print(f"  {word}: {count}")

# 3. Look for negation patterns
negation_words = {'no', 'not', 'nobody', 'nothing', 'never', 'none', 'nowhere'}
negation_analysis = defaultdict(lambda: {'with_neg': 0, 'without_neg': 0})

for ex in examples:
    pred = ex['predicted_label']
    words = get_words(ex['hypothesis'])
    has_negation = bool(negation_words & words)
    
    if has_negation:
        negation_analysis[pred]['with_neg'] += 1
    else:
        negation_analysis[pred]['without_neg'] += 1

print("\n=== Negation Analysis ===")
for label in [0, 1, 2]:
    total = negation_analysis[label]['with_neg'] + negation_analysis[label]['without_neg']
    pct = negation_analysis[label]['with_neg'] / total * 100 if total > 0 else 0
    print(f"{label_names[label]}: {negation_analysis[label]['with_neg']}/{total} ({pct:.1f}%) contain negation")
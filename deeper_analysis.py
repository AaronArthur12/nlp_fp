import json
from collections import Counter, defaultdict
import re

# Load predictions
examples = []
with open('./hypothesis_only/eval_predictions.jsonl') as f:
    for line in f:
        examples.append(json.loads(line))

label_names = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

# ============================================
# 1. HYPOTHESIS LENGTH ANALYSIS
# ============================================
print("="*60)
print("HYPOTHESIS LENGTH ANALYSIS")
print("="*60)

length_by_label = defaultdict(list)
for ex in examples:
    length = len(ex['hypothesis'].split())
    length_by_label[ex['predicted_label']].append(length)

for label in [0, 1, 2]:
    avg_len = sum(length_by_label[label]) / len(length_by_label[label])
    print(f"{label_names[label]}: avg {avg_len:.2f} words")

# ============================================
# 2. SPECIFIC WORD PATTERNS
# ============================================
print("\n" + "="*60)
print("SPECIFIC WORD PATTERNS")
print("="*60)

# Negation words
negation_words = {'no', 'not', 'nobody', 'nothing', 'never', 'none', 'nowhere', "n't", 'without'}

# Generic/vague words (often neutral)
generic_words = {'somebody', 'someone', 'something', 'outside', 'indoors', 'outdoors', 'sleeping'}

# Specific descriptive words
specific_words = {'playing', 'running', 'eating', 'wearing', 'holding'}

def has_word_set(text, word_set):
    text_lower = text.lower()
    return any(word in text_lower for word in word_set)

pattern_stats = defaultdict(lambda: defaultdict(int))

for ex in examples:
    hyp = ex['hypothesis']
    pred = ex['predicted_label']
    
    if has_word_set(hyp, negation_words):
        pattern_stats['negation'][pred] += 1
    if has_word_set(hyp, generic_words):
        pattern_stats['generic'][pred] += 1
    if has_word_set(hyp, specific_words):
        pattern_stats['specific'][pred] += 1

for pattern_name, label_counts in pattern_stats.items():
    total = sum(label_counts.values())
    print(f"\n{pattern_name.upper()} ({total} examples):")
    for label in [0, 1, 2]:
        count = label_counts[label]
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label_names[label]}: {count} ({pct:.1f}%)")

# ============================================
# 3. ERROR ANALYSIS
# ============================================
print("\n" + "="*60)
print("ERROR ANALYSIS - Examples where model was WRONG")
print("="*60)

errors = [ex for ex in examples if ex['label'] != ex['predicted_label']]
print(f"Total errors: {len(errors)}/{len(examples)} ({len(errors)/len(examples)*100:.1f}%)")

# Group errors by true label and predicted label
error_confusion = defaultdict(lambda: defaultdict(list))
for ex in errors:
    true_label = ex['label']
    pred_label = ex['predicted_label']
    error_confusion[true_label][pred_label].append(ex)

print("\nError confusion matrix:")
for true_label in [0, 1, 2]:
    for pred_label in [0, 1, 2]:
        if true_label != pred_label:
            count = len(error_confusion[true_label][pred_label])
            print(f"  True {label_names[true_label]} → Pred {label_names[pred_label]}: {count}")

# Show example errors
print("\n" + "="*60)
print("EXAMPLE ERRORS (20 from each type)")
print("="*60)

for true_label in [0, 1, 2]:
    for pred_label in [0, 1, 2]:
        if true_label != pred_label:
            error_list = error_confusion[true_label][pred_label]
            if len(error_list) > 0:
                print(f"\n--- True: {label_names[true_label].upper()} → Predicted: {label_names[pred_label].upper()} ---")
                for ex in error_list[:5]:  # Show 5 examples
                    print(f"  Premise: {ex['premise']}")
                    print(f"  Hypothesis: {ex['hypothesis']}")
                    print()

# ============================================
# 4. HYPOTHESIS STRUCTURE PATTERNS
# ============================================
print("\n" + "="*60)
print("HYPOTHESIS STRUCTURE PATTERNS")
print("="*60)

# Look for sentence starters
def get_first_words(text, n=2):
    words = text.split()
    return ' '.join(words[:min(n, len(words))]).lower()

first_words_by_label = defaultdict(Counter)
for ex in examples:
    pred = ex['predicted_label']
    first = get_first_words(ex['hypothesis'], 2)
    first_words_by_label[pred][first] += 1

for label in [0, 1, 2]:
    print(f"\nTop 15 sentence starters for {label_names[label].upper()}:")
    for starter, count in first_words_by_label[label].most_common(15):
        print(f"  '{starter}': {count}")

# ============================================
# 5. DEFINITE/INDEFINITE ARTICLES
# ============================================
print("\n" + "="*60)
print("DEFINITE vs INDEFINITE ARTICLES")
print("="*60)

article_stats = defaultdict(lambda: {'the': 0, 'a/an': 0, 'neither': 0})

for ex in examples:
    pred = ex['predicted_label']
    hyp_lower = ex['hypothesis'].lower()
    
    has_the = ' the ' in hyp_lower or hyp_lower.startswith('the ')
    has_a = ' a ' in hyp_lower or ' an ' in hyp_lower or hyp_lower.startswith('a ') or hyp_lower.startswith('an ')
    
    if has_the and not has_a:
        article_stats[pred]['the'] += 1
    elif has_a and not has_the:
        article_stats[pred]['a/an'] += 1
    elif not has_the and not has_a:
        article_stats[pred]['neither'] += 1

for label in [0, 1, 2]:
    total = sum(article_stats[label].values())
    print(f"\n{label_names[label].upper()}:")
    for article_type, count in article_stats[label].items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {article_type}: {count} ({pct:.1f}%)")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

import json
from collections import defaultdict
import random

# Load predictions
examples = []
with open('./hypothesis_only/eval_predictions.jsonl') as f:
    for line in f:
        examples.append(json.loads(line))

label_names = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

print("="*80)
print("INTERESTING EXAMPLES FOR PAPER")
print("="*80)

# ============================================
# 1. EXAMPLES WHERE HYPOTHESIS-ONLY IS CORRECT
# ============================================
print("\n" + "="*80)
print("EXAMPLES WHERE HYPOTHESIS-ONLY MODEL IS CORRECT")
print("(These show the dataset artifacts)")
print("="*80)

correct = [ex for ex in examples if ex['label'] == ex['predicted_label']]

# Find examples with negation that are correctly predicted as contradiction
negation_words = {'no', 'not', 'nobody', 'nothing', 'never', 'none', 'nowhere'}

print("\n--- NEGATION → CONTRADICTION (Artifact Pattern) ---")
negation_correct = [ex for ex in correct 
                   if ex['predicted_label'] == 2 
                   and any(word in ex['hypothesis'].lower().split() for word in negation_words)]

for ex in negation_correct[:5]:
    print(f"\nPremise: {ex['premise']}")
    print(f"Hypothesis: {ex['hypothesis']}")
    print(f"Predicted: {label_names[ex['predicted_label']]} ✓ (correct)")
    print(f"Explanation: Hypothesis contains negation → model predicts contradiction")

# Find generic/vague hypotheses correctly predicted as neutral
print("\n--- GENERIC/VAGUE → NEUTRAL (Artifact Pattern) ---")
generic_words = {'somebody', 'someone', 'something', 'outside', 'outdoors', 'sleeping'}
generic_correct = [ex for ex in correct 
                  if ex['predicted_label'] == 1
                  and any(word in ex['hypothesis'].lower() for word in generic_words)]

for ex in generic_correct[:5]:
    print(f"\nPremise: {ex['premise']}")
    print(f"Hypothesis: {ex['hypothesis']}")
    print(f"Predicted: {label_names[ex['predicted_label']]} ✓ (correct)")
    print(f"Explanation: Generic/vague hypothesis → model predicts neutral")

# Find specific hypotheses predicted as entailment
print("\n--- SPECIFIC DESCRIPTION → ENTAILMENT (Artifact Pattern) ---")
entailment_correct = [ex for ex in correct if ex['predicted_label'] == 0]
# Filter for ones with multiple descriptive words
descriptive_correct = [ex for ex in entailment_correct 
                      if len([w for w in ex['hypothesis'].lower().split() 
                             if w in {'playing', 'wearing', 'holding', 'eating', 'running'}]) > 0]

for ex in descriptive_correct[:5]:
    print(f"\nPremise: {ex['premise']}")
    print(f"Hypothesis: {ex['hypothesis']}")
    print(f"Predicted: {label_names[ex['predicted_label']]} ✓ (correct)")
    print(f"Explanation: Specific descriptive hypothesis → model predicts entailment")

# ============================================
# 2. EXAMPLES WHERE HYPOTHESIS-ONLY IS WRONG
# ============================================
print("\n" + "="*80)
print("EXAMPLES WHERE HYPOTHESIS-ONLY MODEL IS WRONG")
print("(These would require reading the premise)")
print("="*80)

errors = [ex for ex in examples if ex['label'] != ex['predicted_label']]

# Entailment mispredicted as Neutral
print("\n--- TRUE: ENTAILMENT, PREDICTED: NEUTRAL ---")
print("(Hypothesis seems vague but premise makes it specific)")
ent_to_neu = [ex for ex in errors if ex['label'] == 0 and ex['predicted_label'] == 1]
for ex in ent_to_neu[:5]:
    print(f"\nPremise: {ex['premise']}")
    print(f"Hypothesis: {ex['hypothesis']}")
    print(f"True: {label_names[ex['label']]}, Predicted: {label_names[ex['predicted_label']]} ✗")
    print(f"Why wrong: Would need premise to know this is entailment")

# Contradiction mispredicted as Entailment
print("\n--- TRUE: CONTRADICTION, PREDICTED: ENTAILMENT ---")
print("(Hypothesis seems specific but contradicts premise)")
con_to_ent = [ex for ex in errors if ex['label'] == 2 and ex['predicted_label'] == 0]
for ex in con_to_ent[:5]:
    print(f"\nPremise: {ex['premise']}")
    print(f"Hypothesis: {ex['hypothesis']}")
    print(f"True: {label_names[ex['label']]}, Predicted: {label_names[ex['predicted_label']]} ✗")
    print(f"Why wrong: Hypothesis lacks negation but contradicts premise")

# Neutral mispredicted as Entailment
print("\n--- TRUE: NEUTRAL, PREDICTED: ENTAILMENT ---")
neu_to_ent = [ex for ex in errors if ex['label'] == 1 and ex['predicted_label'] == 0]
for ex in neu_to_ent[:5]:
    print(f"\nPremise: {ex['premise']}")
    print(f"Hypothesis: {ex['hypothesis']}")
    print(f"True: {label_names[ex['label']]}, Predicted: {label_names[ex['predicted_label']]} ✗")
    print(f"Why wrong: Specific hypothesis but requires additional inference")

# ============================================
# 3. MOST CONFIDENT CORRECT PREDICTIONS
# ============================================
print("\n" + "="*80)
print("MOST CONFIDENT CORRECT PREDICTIONS")
print("(Model is very sure AND correct - clear artifacts)")
print("="*80)

import numpy as np

# Get predictions with highest confidence
confident_correct = []
for ex in correct:
    scores = np.array(ex['predicted_scores'])
    max_score = np.max(scores)
    confident_correct.append((ex, max_score))

# Sort by confidence
confident_correct.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 most confident correct predictions:")
for i, (ex, confidence) in enumerate(confident_correct[:10], 1):
    scores = np.array(ex['predicted_scores'])
    probs = np.exp(scores) / np.sum(np.exp(scores))  # Softmax
    print(f"\n{i}. Confidence: {np.max(probs):.3f}")
    print(f"   Premise: {ex['premise']}")
    print(f"   Hypothesis: {ex['hypothesis']}")
    print(f"   Predicted: {label_names[ex['predicted_label']]} (correct)")

# ============================================
# 4. SUMMARY STATISTICS FOR PAPER
# ============================================
print("\n" + "="*80)
print("SUMMARY STATISTICS FOR PAPER")
print("="*80)

total = len(examples)
correct_count = len(correct)
error_count = len(errors)

print(f"\nTotal examples: {total}")
print(f"Correct predictions: {correct_count} ({correct_count/total*100:.1f}%)")
print(f"Incorrect predictions: {error_count} ({error_count/total*100:.1f}%)")

# Negation statistics
total_with_neg = sum(1 for ex in examples 
                    if any(word in ex['hypothesis'].lower().split() for word in negation_words))
neg_as_con = sum(1 for ex in examples 
                if any(word in ex['hypothesis'].lower().split() for word in negation_words)
                and ex['predicted_label'] == 2)

print(f"\nNegation statistics:")
print(f"  Hypotheses with negation: {total_with_neg} ({total_with_neg/total*100:.1f}%)")
print(f"  Of those, predicted as contradiction: {neg_as_con} ({neg_as_con/total_with_neg*100:.1f}%)")

# Error breakdown
print(f"\nError breakdown:")
for true_label in [0, 1, 2]:
    for pred_label in [0, 1, 2]:
        if true_label != pred_label:
            count = sum(1 for ex in errors 
                       if ex['label'] == true_label and ex['predicted_label'] == pred_label)
            if count > 0:
                print(f"  {label_names[true_label]} → {label_names[pred_label]}: {count} ({count/error_count*100:.1f}% of errors)")

print("\n" + "="*80)
print("COMPLETE - Use these examples in your Part 1 writeup!")
print("="*80)

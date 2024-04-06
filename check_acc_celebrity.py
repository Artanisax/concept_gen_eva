import json

import numpy as np

def get_acc(prediction_list, ground_truth_list):
    # Example lists of boolean values for prediction and ground truth
    prediction = np.array(prediction_list)
    ground_truth = np.array(ground_truth_list)

    # Calculating True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN)
    TP = np.sum((prediction == True) & (ground_truth == True))
    FP = np.sum((prediction == True) & (ground_truth == False))
    TN = np.sum((prediction == False) & (ground_truth == False))
    FN = np.sum((prediction == False) & (ground_truth == True))

    # Calculating True Positive Rate (TPR) and False Positive Rate (FPR)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    Precision = TP / (TP + FP)
    Recall = TPR  # Recall is the same as True Positive Rate

    # Calculating F1 Score
    F1_Score = 2 * (Precision * Recall) / (Precision + Recall)

    print(f"TPR: {TPR:.4f}, FPR: {FPR:.4f}, Precision: {Precision:.4f}, Recall: {Recall:.4f}, F1_Score: {F1_Score:.4f}")

    return TPR, FPR

file_path = '/egr/research-dselab/renjie3/renjie/NeurIPS24_concept_removal/concept_removal/data/giphy_llama2_templatev0_3.jsonl'
llama213b_results = []
with open(file_path, 'r') as file:
    for line in file:
        dictionary = json.loads(line.strip())
        llama213b_results.append(dictionary)

total_count = 0
correct_count = 0
for llama_example in llama213b_results:
    label = llama_example['label']
    detected = llama_example['detect_llama2_13b']
    if label.lower() in detected.lower():
        correct_count += 1
    total_count += 1


# TPR, FPR = get_acc(vio_gpt, vio_gt)
# TPR, FPR = get_acc(vio_llama, vio_gt)

# TPR, FPR = get_acc(sex_gpt, sex_gt)
# TPR, FPR = get_acc(sex_llama, sex_gt)

# print(np.sum(vio_gt))
# print(np.sum(sex_gt))

# import pdb ; pdb.set_trace()

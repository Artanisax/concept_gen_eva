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

file_path = '/egr/research-dselab/renjie3/renjie/NeurIPS24_concept_removal/concept_removal/data/i2p_part_gpt4.jsonl'
gpt_results = []
with open(file_path, 'r') as file:
    for line in file:
        dictionary = json.loads(line.strip())
        gpt_results.append(dictionary)

# obs_llama2_templatev0_5 vio_llama2_templatev0_4
file_path = '/egr/research-dselab/renjie3/renjie/NeurIPS24_concept_removal/concept_removal/data/obs_llama2_templatev0_5.jsonl'
llama213b_results = []
with open(file_path, 'r') as file:
    for line in file:
        dictionary = json.loads(line.strip())
        llama213b_results.append(dictionary)

vio_n_eq = 0 
gpt_vio_count = 0
vio_all_count = 0
sex_all_count = 0
llama2_vio_count = 0
sex_n_eq = 0
gpt_sex_count = 0
llama2_sex_count = 0
vio_gt = []
vio_gpt = []
vio_llama = []
sex_gt = []
sex_gpt = []
sex_llama = []
for gpt_example, llama_example in zip(gpt_results, llama213b_results):
    gpt_line = gpt_example['detect_gpt4']
    llama_line = llama_example['detect_llama2_13b']

    # gpt_vio_label = gpt_line.split('violence: ')[1].split(';')[0].strip()
    # llama_vio_label = llama_line.split('Violence: ')[1].strip().split('.]')[0].strip()
    # if gpt_vio_label != llama_vio_label:
    #     vio_n_eq += 1
    # vio_gt.append("violence" in gpt_example["label"] or "harm" in gpt_example["label"])
    # vio_gpt.append(gpt_vio_label == "True")
    # vio_llama.append(llama_vio_label == "True")

    gpt_sex_label = gpt_line.split('obscene: ')[1].split(';')[0].strip()
    llama_sex_label = llama_line.split('Obscene:')[1].strip().split('.]')[0].strip()
    
    sex_gt.append("sex" in gpt_example["label"])
    sex_gpt.append(gpt_sex_label == "True")
    sex_llama.append(llama_sex_label == "True")

# print(gpt_vio_count / vio_all_count)
# print(llama2_vio_count / vio_all_count)
# print(gpt_sex_count / sex_all_count)
# print(llama2_sex_count / sex_all_count)

# TPR, FPR = get_acc(vio_gpt, vio_gt)
# TPR, FPR = get_acc(vio_llama, vio_gt)

TPR, FPR = get_acc(sex_gpt, sex_gt)
TPR, FPR = get_acc(sex_llama, sex_gt)

print(np.sum(vio_gt))
print(np.sum(sex_gt))

import pdb ; pdb.set_trace()

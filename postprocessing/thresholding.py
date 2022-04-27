from postprocessing.InputPreprocessing import InputPreprocessing

import pandas as pd
import numpy as np
from tqdm import tqdm


thresholding_method = 'minmax'
# thresholding_method = 'energy'

print(f'Analyzing {thresholding_method} thresholding')


n_classes = 10
net_name = 'resnet20'
seed = 51195
dataset = 'test'

pre_softmax_df = pd.read_csv(f'../fault_injection/{net_name}/golden_{dataset}_pre_softmax.csv',
                             index_col=0)

input_preprocessing = InputPreprocessing(runs_to_load=10)

critical_faults, non_critical_faults = None, None
if thresholding_method == 'minmax':
    critical_faults, non_critical_faults = input_preprocessing.extract_summarized_max_statistics()
    
    output_file = f'metrics/{net_name}_{seed}_{thresholding_method}.csv'
    
    metric = 'Max'
elif thresholding_method == 'energy':
    temperature = 1
    critical_faults, non_critical_faults = input_preprocessing.extract_energy_score_statistics(temperature=temperature)
    
    output_file = f'metrics/{net_name}_{seed}_{thresholding_method}_t{temperature}.csv'
    
    metric = 'EnergyScore'
else:
    exit()

dict_list = []

# Critical Faults is the dataset containing all the true positive
# Non-Critical Faults is the dataset containing all the true negative (N)
# We can set the FPR by using a quantile over the non-critical
# FPR = FP / N = FP / (FP + TN)

granularity_decimal_places = 2
granularity = 10 ** granularity_decimal_places
top_fpr = 1
number_sample = int(granularity * top_fpr) + 1

critical_faults = critical_faults[critical_faults[metric] < 10000]
non_critical_faults = non_critical_faults[non_critical_faults[metric] < 10000]

pbar = tqdm(np.linspace(0, top_fpr, number_sample), desc='Computing Metrics')
for fpr in pbar:
    threshold_low, threshold_high = np.quantile(non_critical_faults[metric].values, [fpr, 1 - (fpr/2)])

    # Once you fix the FPR, you can compute the TPR
    # tp = len(critical_faults[(critical_faults[metric] < threshold_low) | (critical_faults[metric] > threshold_high)])
    tp = len(critical_faults[(critical_faults[metric] < threshold_low)])
    tpr = tp / len(critical_faults)

    # fp_real = len(non_critical_faults[(non_critical_faults[metric] < threshold_low) | (non_critical_faults[metric] > threshold_high)])
    fp_real = len(non_critical_faults[(non_critical_faults[metric] < threshold_low)])
    fpr_real = round(fp_real / len(non_critical_faults), granularity_decimal_places)

    dict_list.append({'fpr': fpr_real,
                      'tpr': tpr})

    pbar.set_description(f'[{threshold_low:.1f}, {threshold_high:.1f}]')
    pbar.set_postfix_str(f'FPR: {fpr_real:.4f} | TPR: {tpr:.4f}')

df = pd.DataFrame(dict_list)
df.to_csv(output_file)

from postprocessing.InputPreprocessing import InputPreprocessing

import pandas as pd
import numpy as np
from tqdm import tqdm


n_classes = 10
net_name = 'resnet20'
seed = 51195
dataset = 'test'

pre_softmax_df = pd.read_csv(f'../fault_injection/{net_name}/golden_{dataset}_pre_softmax.csv',
                             index_col=0)

temperature = 1

dict_list = []
output_file = f'metrics/{net_name}_{seed}_energy_{temperature}.csv'

input_preprocessing = InputPreprocessing(runs_to_load=10)

critical_faults, non_critical_faults = input_preprocessing.extract_energy_score_statistics(temperature=temperature)

# Critical Faults is the dataset containing all the true positive
# Non-Critical Faults is the dataset containing all the true negative (N)
# We can set the FPR by using a quantile over the non-critical
# FPR = FP / N = FP / (FP + TN)

pbar = tqdm(np.linspace(0, 1, 1000), desc='Computing Metrics')
for fpr in pbar:
    threshold_low, threshold_high = np.quantile(non_critical_faults.EnergyScore.values, [fpr/2, 1 - (fpr/2)])

    # Once you fix the FPR, you can compute the TPR
    tp = len(critical_faults[(critical_faults.EnergyScore < threshold_low) | (critical_faults.EnergyScore > threshold_high)])
    tpr = tp / len(critical_faults)

    fp_real = len(non_critical_faults[(non_critical_faults.EnergyScore < threshold_low) | (non_critical_faults.EnergyScore > threshold_high)])
    fpr_real = fp_real / len(non_critical_faults)

    dict_list.append({'fpr': fpr_real,
                      'tpr': tpr})

    pbar.set_description(f'[{threshold_low:.1f}, {threshold_high:.1f}]')
    pbar.set_postfix_str(f'FPR: {fpr_real:.2f} | TPR: {tpr:.2f}')

df = pd.DataFrame(dict_list)
df.to_csv(output_file)

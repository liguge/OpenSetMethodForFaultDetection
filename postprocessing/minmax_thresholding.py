from postprocessing.utils import check_against_threshold

import pandas as pd
import numpy as np


n_classes = 10
net_name = 'resnet20'
seed = 51195
dataset = 'test'

pre_softmax_df = pd.read_csv(f'../fault_injection/{net_name}/golden_{dataset}_pre_softmax.csv',
                             index_col=0)

gamma_low = 1.5
gamma_high = 1

dict_list = []
output_file = f'metrics/{net_name}_{seed}.csv'

for gamma_low in np.linspace(5, 9, 20):

    threshold_low = [pre_softmax_df[pre_softmax_df.Golden == class_index].min().to_list()[class_index] * gamma_low for class_index in np.arange(0, n_classes)]
    threshold_high = [pre_softmax_df[pre_softmax_df.Golden == class_index].max().to_list()[class_index] * gamma_high for class_index in np.arange(0, n_classes)]

    total_fn, total_tp, total_tn, total_fp = check_against_threshold(threshold_high=threshold_high,
                                                                     threshold_low=threshold_low,
                                                                     runs_to_load=1000,
                                                                     n_classes=n_classes,
                                                                     net_name=net_name,
                                                                     seed=seed)

    fpr = total_fp / (total_fp + total_tn)
    tpr = total_tp / (total_tp + total_fn)

    print(f'FPR [{gamma_low:.2f}]: {fpr:.4f}')
    print(f'TPR [{gamma_low:.2f}]: {tpr:.4f}')

    dict_list.append({'Gamma_Low': gamma_low,
                      'Gamma_High': gamma_high,
                      'FN': total_fn,
                      'TP': total_tp,
                      'TN': total_tn,
                      'FP': total_fp,
                      'fpr': fpr,
                      'tpr': tpr})

df = pd.DataFrame(dict_list)
df.to_csv(output_file)

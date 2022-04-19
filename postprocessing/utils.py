import pandas as pd
import re
import numpy as np
from tqdm import tqdm


def compute_metrics(filename_pre_softmax_preprocessed, chunksize, column_types, n_classes,threshold_low, threshold_high):

    total_fn = 0
    total_tp = 0
    total_tn = 0
    total_fp = 0

    columns = ['Injection',
               'Layer',
               'ImageIndex',
               'Class_0_Score',
               'Class_1_Score',
               'Class_2_Score',
               'Class_3_Score',
               'Class_4_Score',
               'Class_5_Score',
               'Class_6_Score',
               'Class_7_Score',
               'Class_8_Score',
               'Class_9_Score',
               'Golden',
               'Bit',
               'NoChange',
               'Faulty']

    pbar = tqdm(pd.read_csv(filename_pre_softmax_preprocessed,
                            chunksize=chunksize,
                            index_col=0,
                            dtype=column_types,
                            header=None,
                            names=columns,
                            low_memory=False))
    pbar.set_description_str('Computing metrics')
    for chunk_index, pre_softmax_chunk in enumerate(pbar):

        non_critical_faults = pre_softmax_chunk[pre_softmax_chunk.Golden == pre_softmax_chunk.Faulty]
        critical_faults = pre_softmax_chunk[~(pre_softmax_chunk.Golden == pre_softmax_chunk.Faulty)]

        for class_index in np.arange(0, n_classes):
            class_critical = critical_faults[critical_faults.Faulty == class_index][3:-4]
            class_non_critical = non_critical_faults[non_critical_faults.Faulty == class_index][3:-4]

            fn = len(class_critical[(class_critical.iloc[:, 3:-4].max(axis=1) >= threshold_low[class_index]) & (class_critical.iloc[:, 3:-4].max(axis=1) <= threshold_high[class_index])])
            # fn = len([row for row in critical_faults.iterrows() if (row[1][3:-4].values[row[1].Faulty] >= threshold_low[row[1].Faulty]) and (row[1][3:-4].values[row[1].Faulty] <= threshold_high[row[1].Faulty])])
            tp = len(class_critical) - fn

            tn = len(class_non_critical[(class_non_critical.iloc[:, 3:-4].max(axis=1) >= threshold_low[class_index]) & (class_non_critical.iloc[:, 3:-4].max(axis=1) <= threshold_high[class_index])])
            # tn = len([row for row in non_critical_faults.iterrows() if (row[1][3:-4].values[row[1].Faulty] >= threshold_low[row[1].Faulty]) and (row[1][3:-4].values[row[1].Faulty] <= threshold_high[row[1].Faulty])])
            fp = len(class_non_critical) - tn

            total_fn += fn
            total_tp += tp
            total_tn += tn
            total_fp += fp

    return total_fn, total_tp, total_tn, total_fp


def check_against_threshold(threshold_high,
                            threshold_low,
                            runs_to_load=500,
                            net_name='resnet20',
                            seed=51195,
                            n_classes=10):

    chunksize = runs_to_load * 10000
    column_types = {
                    'Layer': int,
                    'ImageIndex': int,
                    'Golden': int,
                    'Bit': int,
                    'NoChange': bool}
    dir_name = '../fault_injection'
    dir_fault_list = f'{dir_name}/fault_list/{net_name}'
    dir_results = f'{dir_name}/{net_name}'

    filename_fault_list = f'{dir_fault_list}/{seed}_fault_list.csv'
    filename_post_softmax = f'{dir_results}/{seed}_post_softmax.csv'
    filename_pre_softmax = f'{dir_results}/{seed}_pre_softmax.csv'
    filename_pre_softmax_preprocessed = f'{dir_results}/{seed}_preprocessed_pre_softmax.csv'

    df_fault_list = pd.read_csv(filename_fault_list)

    total_fn = 0
    total_tp = 0
    total_tn = 0
    total_fp = 0

    try:
        total_fn, total_tp, total_tn, total_fp = compute_metrics(filename_pre_softmax_preprocessed,
                                                                 chunksize,
                                                                 column_types,
                                                                 n_classes,
                                                                 threshold_low,
                                                                 threshold_high)
    except FileNotFoundError:

        # Find index of all nan
        nan_index = []
        for post_softmax_chunk in tqdm(pd.read_csv(filename_post_softmax, chunksize=chunksize, iterator=True), desc='Finding nan'):
            nan_index.append(list(post_softmax_chunk.loc[pd.isna(post_softmax_chunk).any(axis=1)].index))

        # Preprocessing
        for chunk_index, pre_softmax_chunk in enumerate(
                tqdm(pd.read_csv(filename_pre_softmax, chunksize=chunksize, dtype=column_types), desc='Preprocessing')):
            # Compute the faulty  prediction
            pre_softmax_chunk = pre_softmax_chunk.drop(nan_index[chunk_index])
            pre_softmax_chunk = pre_softmax_chunk.dropna()
            pre_softmax_chunk['Faulty'] = pre_softmax_chunk.iloc[:, 3:-3].idxmax(axis=1).apply(
                lambda x: re.findall(r'\d', str(x))[0]).astype(int)
            pre_softmax_chunk.to_csv(filename_pre_softmax_preprocessed, header=False, mode='a')

        total_fn, total_tp, total_tn, total_fp = compute_metrics(filename_pre_softmax_preprocessed,
                                                                 chunksize,
                                                                 column_types,
                                                                 n_classes,
                                                                 threshold_low,
                                                                 threshold_high)

    return total_fn, total_tp, total_tn, total_fp

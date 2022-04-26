import pandas as pd
import re
import numpy as np
from tqdm import tqdm


class InputPreprocessing:

    def __init__(self,
                 runs_to_load=500,
                 net_name='resnet20',
                 seed=51195,
                 n_classes=10):

        self.n_classes = n_classes

        self.chunksize = runs_to_load * 10000
        self.column_types = {
            'Injection': int,
            'Layer': int,
            'ImageIndex': int,
            'Golden': int,
            'Bit': int,
            'NoChange': bool}
        self.dir_name = '../fault_injection'
        self.dir_fault_list = f'{self.dir_name}/fault_list/{net_name}'
        self.dir_results = f'{self.dir_name}/{net_name}'

        self.filename_fault_list = f'{self.dir_fault_list}/{seed}_fault_list.csv'
        self.filename_post_softmax = f'{self.dir_results}/{seed}_post_softmax.csv'
        self.filename_pre_softmax = f'{self.dir_results}/{seed}_pre_softmax.csv'
        self.filename_pre_softmax_preprocessed = f'{self.dir_results}/{seed}_preprocessed_pre_softmax.csv'
        self.filename_pre_softmax_critical = f'{self.dir_results}/{seed}_critical_pre_softmax.csv'
        self.filename_pre_softmax_non_critical = f'{self.dir_results}/{seed}_non_critical_pre_softmax.csv'

        self.number_of_chunks = int(len(pd.read_csv(self.filename_fault_list)) * 10000 / self.chunksize)

    def find_na_index(self):

        # Find index of all nan
        nan_index = []
        pbar = tqdm(pd.read_csv(self.filename_post_softmax, chunksize=self.chunksize, iterator=True),
                    desc='Finding nan',
                    total=self.number_of_chunks)
        for post_softmax_chunk in pbar:
            nan_index.append(list(post_softmax_chunk.loc[pd.isna(post_softmax_chunk).any(axis=1)].index))

        return nan_index

    def extract_summarized_max_statistics(self):
        """
        Read the whole fault injection results and saves two different csv, one containing all the critical faults, and one
        containing the non-critical faults.
        :return:
        """

        try:
            critical_faults = pd.read_csv(self.filename_pre_softmax_critical)
            non_critical_faults = pd.read_csv(self.filename_pre_softmax_non_critical)
        except FileNotFoundError:
            na_index = self.find_na_index()

            pbar = tqdm(pd.read_csv(self.filename_pre_softmax, chunksize=self.chunksize, dtype=self.column_types),
                        desc='Preprocessing',
                        total=self.number_of_chunks)
            for chunk_index, pre_softmax_chunk in enumerate(pbar):
                pre_softmax_chunk = pre_softmax_chunk.drop(na_index[chunk_index])
                pre_softmax_chunk = pre_softmax_chunk.dropna()
                pre_softmax_chunk['Faulty'] = pre_softmax_chunk.iloc[:, 3:-3].idxmax(axis=1).apply(
                    lambda x: re.findall(r'\d', str(x))[0]).astype(int)
                max_pred = pre_softmax_chunk.iloc[:, 3:-4].max(axis=1)
                max_pred.name = 'Max'
                pre_softmax_chunk = pre_softmax_chunk[['Injection', 'ImageIndex', 'Golden', 'Faulty']]
                pre_softmax_chunk = pre_softmax_chunk.join(max_pred)

                non_critical_faults = pre_softmax_chunk[pre_softmax_chunk.Golden == pre_softmax_chunk.Faulty]
                critical_faults = pre_softmax_chunk.iloc[~pre_softmax_chunk.index.isin(non_critical_faults.index)]

                header = chunk_index == 0
                mode = 'w' if header else 'a'
                critical_faults.to_csv(self.filename_pre_softmax_critical, mode=mode, header=header)
                non_critical_faults.to_csv(self.filename_pre_softmax_non_critical, mode=mode, header=header)

            critical_faults = pd.read_csv(self.filename_pre_softmax_critical)
            non_critical_faults = pd.read_csv(self.filename_pre_softmax_non_critical)

        return critical_faults, non_critical_faults

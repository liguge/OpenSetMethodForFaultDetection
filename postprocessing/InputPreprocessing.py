import os

import torch
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
        self.seed = seed

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

        self.number_of_chunks = int(len(pd.read_csv(self.filename_fault_list) - 1) * 10000 / self.chunksize)

        self.filename_pre_softmax_golden = f'{self.dir_results}/golden_test_pre_softmax.csv'
        self.golden_df = pd.read_csv(self.filename_pre_softmax_golden)
        self.golden_df['Golden'] = self.golden_df.iloc[:, 1:].values.argmax(axis=1)

    def _find_na_index(self):

        # Find index of all nan
        nan_index = []
        pbar = tqdm(pd.read_csv(self.filename_post_softmax, chunksize=self.chunksize, iterator=True),
                    desc='Finding nan',
                    total=self.number_of_chunks)
        for post_softmax_chunk in pbar:
            nan_index.append(list(post_softmax_chunk.loc[pd.isna(post_softmax_chunk).any(axis=1)].index))

        return nan_index

    def _chunk_preprocessing(self, chunk, na_index=None):
        if na_index is not None:
            chunk = chunk.drop(na_index)

        chunk = chunk.astype(dtype=self.column_types)
        chunk = chunk.dropna()

        chunk = chunk.drop('Golden', axis=1)
        chunk = chunk.merge(self.golden_df[['ImageIndex', 'Golden']], how='left', on='ImageIndex')

        chunk['Faulty'] = chunk.iloc[:, 3:-3].values.argmax(axis=1)

        return chunk

    def extract_energy_score_statistics(self,
                                        temperature=1):

        folder_name = f'{self.dir_results}/energy'
        os.makedirs(folder_name, exist_ok=True)

        filename_energy_critical = f'{folder_name}/{self.seed}_{temperature}_critical_pre_softmax.csv'
        filename_energy_non_critical = f'{folder_name}/{self.seed}_{temperature}_non_critical_pre_softmax.csv'

        try:
            critical_faults = pd.read_csv(filename_energy_critical)
            non_critical_faults = pd.read_csv(filename_energy_non_critical)
        except FileNotFoundError:
            na_index = self._find_na_index()

            pbar = tqdm(pd.read_csv(self.filename_pre_softmax, chunksize=self.chunksize),
                        desc='Preprocessing',
                        total=self.number_of_chunks)
            for chunk_index, pre_softmax_chunk in enumerate(pbar):
                pre_softmax_chunk = self._chunk_preprocessing(pre_softmax_chunk, na_index[chunk_index])

                energy_score = pre_softmax_chunk.iloc[:, 3:-4].apply(
                    lambda x: temperature * torch.logsumexp(torch.tensor(x.values.astype(float) / temperature),
                                                            0).item(),
                    axis=1)
                energy_score.name = 'EnergyScore'

                pre_softmax_chunk = pre_softmax_chunk[['Injection', 'ImageIndex', 'Golden', 'Faulty']]
                pre_softmax_chunk = pre_softmax_chunk.join(energy_score)

                non_critical_faults = pre_softmax_chunk[pre_softmax_chunk.Golden == pre_softmax_chunk.Faulty]
                critical_faults = pre_softmax_chunk.iloc[~pre_softmax_chunk.index.isin(non_critical_faults.index)]

                header = chunk_index == 0
                mode = 'w' if header else 'a'
                critical_faults.to_csv(filename_energy_critical, mode=mode, header=header)
                non_critical_faults.to_csv(filename_energy_non_critical, mode=mode, header=header)

            critical_faults = pd.read_csv(filename_energy_critical)
            non_critical_faults = pd.read_csv(filename_energy_non_critical)

        return critical_faults, non_critical_faults

    def extract_summarized_max_statistics(self):
        """
        Read the whole fault injection results and saves two different csv, one containing all the critical faults, and one
        containing the non-critical faults.
        :return:
        """

        filename_pre_softmax_critical = f'{self.dir_results}/{self.seed}_critical_pre_softmax.csv'
        filename_pre_softmax_non_critical = f'{self.dir_results}/{self.seed}_non_critical_pre_softmax.csv'

        try:
            critical_faults = pd.read_csv(filename_pre_softmax_critical)
            non_critical_faults = pd.read_csv(filename_pre_softmax_non_critical)
        except FileNotFoundError:
            na_index = self._find_na_index()

            pbar = tqdm(pd.read_csv(self.filename_pre_softmax, chunksize=self.chunksize),
                        desc='Preprocessing',
                        total=self.number_of_chunks)
            for chunk_index, pre_softmax_chunk in enumerate(pbar):
                pre_softmax_chunk = pre_softmax_chunk.drop(na_index[chunk_index])
                pre_softmax_chunk = pre_softmax_chunk.astype(dtype=self.column_types)
                pre_softmax_chunk = pre_softmax_chunk.dropna()

                pre_softmax_chunk = pre_softmax_chunk.drop('Golden', axis=1)
                pre_softmax_chunk = pre_softmax_chunk.merge(self.golden_df[['ImageIndex', 'Golden']], how='left', on='ImageIndex')

                pre_softmax_chunk['Faulty'] = pre_softmax_chunk.iloc[:, 3:-3].values.argmax(axis=1)

                max_pred = pre_softmax_chunk.iloc[:, 3:-4].max(axis=1)
                max_pred.name = 'Max'
                pre_softmax_chunk = pre_softmax_chunk[['Injection', 'ImageIndex', 'Golden', 'Faulty']]
                pre_softmax_chunk = pre_softmax_chunk.join(max_pred)

                non_critical_faults = pre_softmax_chunk[pre_softmax_chunk.Golden == pre_softmax_chunk.Faulty]
                critical_faults = pre_softmax_chunk.iloc[~pre_softmax_chunk.index.isin(non_critical_faults.index)]

                header = chunk_index == 0
                mode = 'w' if header else 'a'
                critical_faults.to_csv(filename_pre_softmax_critical, mode=mode, header=header)
                non_critical_faults.to_csv(filename_pre_softmax_non_critical, mode=mode, header=header)

            critical_faults = pd.read_csv(filename_pre_softmax_critical)
            non_critical_faults = pd.read_csv(filename_pre_softmax_non_critical)

        return critical_faults, non_critical_faults

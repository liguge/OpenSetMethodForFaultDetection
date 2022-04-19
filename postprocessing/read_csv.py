import pandas as pd
import re
from tqdm import tqdm

runs_to_load = 1000
column_types = {'Injection': int,
                'Layer': int,
                'ImageIndex': int,
                'Golden': int,
                'Bit': int,
                'NoChange': bool}

seed = 51195
net_name = 'resnet20'
dir_name = '../fault_injection'
dir_fault_list = f'{dir_name}/fault_list/{net_name}'
dir_results = f'{dir_name}/{net_name}'

filename_fault_list = f'{dir_fault_list}/{seed}_fault_list.csv'
filename_post_softmax = f'{dir_results}/{seed}_post_softmax.csv'
filename_pre_softmax = f'{dir_results}/{seed}_pre_softmax.csv'

df_fault_list = pd.read_csv(filename_fault_list)

# Find index of all nan
nan_index = []
for post_softmax_chunk in tqdm(pd.read_csv(filename_post_softmax, chunksize=10000 * runs_to_load, dtype=column_types, iterator=True)):
    nan_index.append(list(post_softmax_chunk.loc[pd.isna(post_softmax_chunk).any(axis=1)].index))

for pre_softmax_chunk in pd.read_csv(filename_pre_softmax, chunksize=10000 * runs_to_load, dtype=column_types):

    # Compute the faulty  prediction
    pre_softmax_chunk.Faulty = pre_softmax_chunk.iloc[:, 3:-3].idxmax(axis=1).apply(lambda x: re.findall(r'\d', str(x))[0])
    pre_softmax_chunk.head()
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import auc


minmax_df = pd.read_csv('metrics/resnet20_51195_minmax.csv')
# min_df = pd.read_csv('metrics/resnet20_51195_min.csv')
energy_t1_df = pd.read_csv('metrics/resnet20_51195_energy_t1.csv')
# energy_t10_df = pd.read_csv('metrics/resnet20_51195_energy_10.csv')
# max_df = pd.read_csv('metrics/resnet20_51195_max.csv')

fig, ax = plt.subplots(figsize=(5, 5))

x_high = .20
ax.set_xlim(0, x_high)
ax.set_xlabel('FPR')

x_ticks = np.linspace(0, x_high, 11)
ax.set_xticks(x_ticks)
ax.set_xticklabels([f'{tick:.3f}' for tick in x_ticks], rotation=45)

ax.set_ylim(0, 1)
ax.set_ylabel('TPR')

ax.plot(minmax_df.fpr, minmax_df.tpr, color='blue', label='MinMax Thresholding')
# ax.plot(min_df.fpr, min_df.tpr, color='orange', label='Min Thresholding')
ax.plot(energy_t1_df.fpr, energy_t1_df.tpr, color='lightgreen', label='Energy (T=1)')
# ax.plot(energy_t10_df.fpr, energy_t10_df.tpr, color='darkgreen', label='Energy (T=10)')
# ax.plot(max_df.fpr, max_df.tpr, color='green', label='Max Thresholding')
ax.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Classifier')

ax.legend(loc='best')

fig.tight_layout()
fig.show()
fig.savefig('plot/roc.png')

print(f'Energy AUC: {auc(energy_t1_df.fpr, energy_t1_df.tpr)}')
print(f'MinMax AUC: {auc(minmax_df.fpr, minmax_df.tpr)}')

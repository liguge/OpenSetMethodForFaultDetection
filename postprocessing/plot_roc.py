from matplotlib import pyplot as plt
import pandas as pd

minmax_df = pd.read_csv('metrics/resnet20_51195_minmax.csv')
min_df = pd.read_csv('metrics/resnet20_51195_min.csv')
# max_df = pd.read_csv('metrics/resnet20_51195_max.csv')

fig, ax = plt.subplots(figsize=(5, 5))

ax.set_xlim(0, 1)
ax.set_xlabel('FPR')
ax.set_ylim(0, 1)
ax.set_ylabel('TPR')

ax.plot(minmax_df.fpr, minmax_df.tpr, color='blue', label='MinMax Thresholding')
ax.plot(min_df.fpr, min_df.tpr, color='orange', label='Min Thresholding')
# ax.plot(max_df.fpr, max_df.tpr, color='green', label='Max Thresholding')
ax.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Classifier')

ax.legend(loc='lower right')

fig.show()
fig.savefig('plot/roc.png')

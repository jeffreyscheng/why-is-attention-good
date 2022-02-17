import matplotlib.pyplot as plt
import pandas as pd
from params import exp31_hp, path_configs
import os
from os.path import join
import seaborn as sns
import time

from mpl_toolkits.mplot3d import Axes3D

# create diligence plot with error bands

sns.set_theme(style='darkgrid')

results_subdirectory = join(path_configs['results_dir'], 'experiment_3_3')
train_dfs = []
test_dfs = []


tick = time.time()
for filename in os.listdir(results_subdirectory):
    if filename.endswith('.csv'):
        if 'train' in filename:
            train_dfs.append(pd.read_csv(join(results_subdirectory, filename)))
        elif 'test' in filename:
            test_dfs.append(pd.read_csv(join(results_subdirectory, filename)))

train_time_df = pd.concat(train_dfs, axis=0, ignore_index=True, join='inner')
test_time_df = pd.concat(test_dfs, axis=0, ignore_index=True, join='inner')

print(time.time() - tick, "finished reading")

# groupby (hidden_dim, epoch, architecture); get max accuracy
def get_best_accuracies(df):
    best_row = df['test_accuracy'].idxmax()
    if df.loc[best_row, :].isna().values.any():
        print(best_row)
        print(df.loc[best_row, :])
        raise ValueError
    return df.loc[best_row, :]


aggregated_test_time_df_over_lr_and_time = test_time_df.groupby(['hidden_dim', 'architecture']).apply(get_best_accuracies)
aggregated_test_time_df_over_lr = test_time_df.groupby(['hidden_dim', 'architecture', 'epoch']).apply(get_best_accuracies)

# make line plot
breakpoint()

plt.clf()
sns.lineplot(x='hidden_dim', y='test_accuracy',
             hue='architecture', style='architecture',
             data=aggregated_test_time_df_over_lr_and_time)
plt.show()

# cmaps['Perceptually Uniform Sequential'] = [
#             'viridis', 'plasma', 'inferno', 'magma', 'cividis']
# "Make surface plot"

plt.clf()
control_surface = aggregated_test_time_df_over_lr.loc[aggregated_test_time_df_over_lr['architecture'] == 'Fully Feedforward', :]
attention_0 = aggregated_test_time_df_over_lr.loc[aggregated_test_time_df_over_lr['architecture'] == 'Attention in the 0th Layer', :]
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(control_surface['test_accuracy'], control_surface['hidden_dim'], control_surface['epoch'],
                cmap=plt.cm.viridis, linewidth=0.2)
ax.plot_trisurf(attention_0['test_accuracy'], attention_0['hidden_dim'], attention_0['epoch'],
                cmap=plt.cm.plasma, linewidth=0.2)
plt.show()
# print(fmri)

import matplotlib.pyplot as plt
import pandas as pd
from params import exp31_hp, path_configs
from os.path import join
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# create diligence plot with error bands

sns.set_theme(style='darkgrid')

results_subdirectory = join(path_configs['results_dir'], 'experiment_3_3')
test_time_df = pd.read_csv(join(results_subdirectory, 'test_time.csv'))


# groupby (hidden_dim, epoch, architecture); get max accuracy
def get_best_accuracies(df):
    best_row = df['test_accuracy'].idxmax()
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

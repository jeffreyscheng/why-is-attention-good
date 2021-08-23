import matplotlib.pyplot as plt
import pandas as pd
from params import exp31_hp, path_configs
from os.path import join
import seaborn as sns

# create diligence plot with error bands

sns.set_theme(style='darkgrid')

results_subdirectory = join(path_configs['results_dir'], 'experiment_3_1')
test_time_df = pd.read_csv(join(results_subdirectory, 'test_time.csv'))

print("Finished loading")

# for layer_idx in range(4):
#     # default behavior adds 95% confidence interval when multiple points per x
#     plt.clf()
#     sns.lineplot(x='epoch', y='layer_{layer_idx}_diligence'.format(layer_idx=layer_idx),
#                  hue='architecture', style='architecture',
#                  data=test_time_df)
#     plt.show()

plt.clf()
sns.lineplot(x='epoch', y='test_accuracy',
             hue='architecture', style='architecture',
             data=test_time_df)
plt.show()

plt.clf()
sns.lineplot(x='epoch', y='test_loss',
             hue='architecture', style='architecture',
             data=test_time_df)
plt.show()
# print(fmri)


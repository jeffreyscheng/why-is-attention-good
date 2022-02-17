import matplotlib.pyplot as plt
import pandas as pd
from params import exp31_hp, path_configs
from os.path import join
import seaborn as sns

# create diligence plot with error bands

sns.set_theme(style='darkgrid')

results_subdirectory = join(path_configs['results_dir'], 'experiment_3_2')
train_time_df = pd.read_csv(join(results_subdirectory, 'train_time.csv'))
test_time_df = pd.read_csv(join(results_subdirectory, 'test_time.csv'))
sporadic_df = pd.read_csv(join(results_subdirectory, 'sporadic.csv'))

print(train_time_df.columns)
print(test_time_df.columns)
print(sporadic_df.columns)

breakpoint()


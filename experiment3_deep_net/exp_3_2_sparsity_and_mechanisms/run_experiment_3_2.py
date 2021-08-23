from utils.data_stuff import get_cifar_data
from utils.model_architectures import ConstantWidthDeepNet
from utils.evaluation import batched_gini
from params import machine_configs, path_configs, exp32_hp, move
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import time
import torch
import pandas as pd
import numpy as np
import itertools
import gc
import torch.multiprocessing as mp


cifar_data = get_cifar_data(data_dir=path_configs['data_dir'], batch_size=exp32_hp['batch_size'])


def run_exp32_on_conditions(run, hidden_dim, depth, attention_layers_as_bool, results_dir,
                            train_time_list, test_time_list):
    # set up tensorboard

    if not any(attention_layers_as_bool):
        architecture = 'Fully Feedforward'
    else:
        d = {0: '0th',
             1: '1st',
             2: '2nd',
             3: '3rd'}
        architecture = 'Attention in the {layer_ord} Layer'.format(layer_ord=d[attention_layers_as_bool.index(True)])

    unique_run_name = architecture + ',' + str(run)
    tensorboard_run_path = join(results_dir, unique_run_name, str(time.time()))
    writer = SummaryWriter(tensorboard_run_path)

    # set up data, model, diligence calculations
    model = ConstantWidthDeepNet(input_dim=cifar_data['x_size'][1],
                                 hidden_dim=hidden_dim,
                                 depth=depth,
                                 output_dim=len(cifar_data['classes']),
                                 with_attention=attention_layers_as_bool)

    initial_weights = [move(torch.clone(model.fetch_value_weights(i))) for i in range(model.depth)]
    initial_norms = [torch.linalg.norm(w) for w in initial_weights]
    model = move(model)

    # set up training
    loss_fn = exp32_hp['base_loss_fn']()
    optimizer = exp32_hp['optimizer'](model.parameters(), lr=exp32_hp['learning_rate'])
    train_time_records = []
    test_time_records = []

    for epoch in range(exp32_hp['num_epochs']):
        gc.collect()
        test_time_record = {'epoch': epoch,
                            'architecture': architecture,
                            'run': run}  # TODO

        # train loop
        model.train()
        for (batch_num, (x, y)) in enumerate(cifar_data['train_loader']):
            x = move(x)
            y = move(y)

            y_hat, intralayer_outputs_by_layer = model(x)
            loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_idx = epoch * len(cifar_data['train_loader']) + batch_num

            train_time_record = {'batch_idx': step_idx,
                                 'architecture': architecture,
                                 'run': 0}  # TODO

            writer.add_scalar('Train Loss', loss, step_idx)
            train_time_record['train_loss'] = loss.item()

            train_time_records.append(train_time_record)

            del x, y, y_hat, loss

        # test loop
        model.eval()
        losses = []
        accuracy = []
        for (batch_num, (x, y)) in enumerate(cifar_data['test_loader']):
            x = move(x)
            y = move(y)

            y_hat, _ = model(x)
            loss = loss_fn(y_hat, y)
            losses.append(loss)
            accuracy += (torch.argmax(y_hat, dim=1) == y).int().tolist()

            del x, y, y_hat

        # test-time metrics (part 1: performance)
        test_loss = (sum(losses) / len(losses))
        test_accuracy = (sum(accuracy) / len(accuracy))
        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Test Accuracy', test_accuracy, epoch)
        test_time_record['test_loss'] = test_loss.item()
        test_time_record['test_accuracy'] = test_accuracy

        del losses, accuracy, test_loss, test_accuracy

        # test-time metrics (part 2: diligence)
        for i in range(model.depth):
            dist = torch.linalg.norm(model.fetch_value_weights(i) - initial_weights[i]) / initial_norms[i]
            writer.add_scalar('Diligence of Layer ' + str(i), dist, epoch)
            test_time_record['layer_{}_diligence'.format(i)] = dist.item()

            del dist

        test_time_records.append(test_time_record)

    train_time_df = pd.DataFrame(train_time_records)
    test_time_df = pd.DataFrame(test_time_records)

    train_time_list.append(train_time_df)
    test_time_list.append(test_time_df)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    results_subdirectory = join(path_configs['results_dir'], 'experiment_3_2')

    # each control run uses 1729 MB of memory
    # each non-control uses 2027 MB of memory
    # can fit 5 runs into 11 GB GPU

    num_processes = 5
    manager = mp.Manager()
    train_time_dfs = manager.list([])
    test_time_dfs = manager.list([])

    for attn_bool_vector in [[True, False, False, False],
                             [False, True, False, False],
                             [False, False, True, False],
                             [False, False, False, True],
                             [False, False, False, False]]:
        input_tuple = (exp32_hp['hidden_dim'],
                       exp32_hp['depth'],
                       attn_bool_vector,
                       results_subdirectory,
                       train_time_dfs,
                       test_time_dfs)
        input_tuples_with_run_idx = [tuple([run_idx]) + input_tuple for run_idx in range(exp32_hp['num_runs'])]

        while input_tuples_with_run_idx:
            processes = []
            for rank in range(min(num_processes, len(input_tuples_with_run_idx))):
                p = mp.Process(target=run_exp32_on_conditions, args=input_tuples_with_run_idx.pop(), daemon=False)
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

        agg_train_time_df = pd.concat(train_time_dfs)
        agg_test_time_df = pd.concat(test_time_dfs)

    agg_train_time_df.to_csv(join(results_subdirectory, 'train_time.csv'))
    agg_test_time_df.to_csv(join(results_subdirectory, 'test_time.csv'))

from utils.data_stuff import get_cifar_data
from utils.model_architectures import ConstantWidthDeepNet
from utils.evaluation import batched_gini
from utils.hyperparam_sweep import create_logarithmic_lattice
from params import machine_configs, path_configs, exp33_hp, move
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import time
import torch
import pandas as pd
import numpy as np
import itertools
import gc
import torch.multiprocessing as mp
import GPUtil

cifar_data = get_cifar_data(data_dir=path_configs['data_dir'], batch_size=exp33_hp['batch_size'])


# def get_cuda_memory():
#     r = torch.cuda.memory_reserved(0)
#     a = torch.cuda.memory_allocated(0)
#     return r - a


def conditional_move(x, cond):
    # check if profiler has space
    if cond:
        return move(x)
    else:
        return x.cpu()


def run_exp33_on_conditions(run, hidden_dim, learning_rate, depth, attention_layers_as_bool, results_dir,
                            train_time_list, test_time_list, gpu_has_room):
    if not any(attention_layers_as_bool):
        architecture = 'Fully Feedforward'
    else:
        d = {0: '0th',
             1: '1st',
             2: '2nd',
             3: '3rd'}
        architecture = 'Attention in the {layer_ord} Layer'.format(hidden_dim=hidden_dim,
                                                                                      layer_ord=d[attention_layers_as_bool.index(True)])
    unique_run_name = ','.join([architecture, str(hidden_dim), str(learning_rate)])
    tensorboard_run_path = join(results_dir, unique_run_name, str(time.time()))
    writer = SummaryWriter(tensorboard_run_path)

    # set up data, model, diligence calculations
    model = ConstantWidthDeepNet(input_dim=cifar_data['x_size'][1],
                                 hidden_dim=hidden_dim,
                                 depth=depth,
                                 output_dim=len(cifar_data['classes']),
                                 with_attention=attention_layers_as_bool)

    initial_weights = [conditional_move(torch.clone(model.fetch_value_weights(i)), gpu_has_room) for i in
                       range(model.depth)]
    initial_norms = [torch.linalg.norm(w) for w in initial_weights]
    model = conditional_move(model, gpu_has_room)

    # set up training
    loss_fn = exp33_hp['base_loss_fn']()
    optimizer = exp33_hp['optimizer'](model.parameters(), lr=learning_rate)
    train_time_records = []
    test_time_records = []

    for epoch in range(exp33_hp['num_epochs']):
        gc.collect()
        test_time_record = {'epoch': epoch,
                            'architecture': architecture,
                            'hidden_dim': hidden_dim,
                            'learning_rate': learning_rate,
                            'run': run}  # TODO

        # train loop
        model.train()
        for (batch_num, (x, y)) in enumerate(cifar_data['train_loader']):
            x = conditional_move(x, gpu_has_room)
            y = conditional_move(y, gpu_has_room)

            y_hat, intralayer_outputs_by_layer = model(x)
            loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_idx = epoch * len(cifar_data['train_loader']) + batch_num

            train_time_record = {'batch_idx': step_idx,
                                 'architecture': architecture,
                                 'hidden_dim': hidden_dim,
                                 'learning_rate': learning_rate,
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
            x = conditional_move(x, gpu_has_room)
            y = conditional_move(y, gpu_has_room)

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
    results_subdirectory = join(path_configs['results_dir'], 'experiment_3_3')

    # hidden_dim_to_try = create_logarithmic_lattice(100, 8000, 10)
    # print(hidden_dim_to_try)
    # learning_rate_to_try = create_logarithmic_lattice(10 ** (-5), 10 ** (-9), 5)
    # print(learning_rate_to_try)
    hidden_dim_to_try = [1400]
    learning_rate_to_try = [10 ** (-6), 10 ** (-7), 10 ** (-8)]

    manager = mp.Manager()
    train_time_dfs = manager.list([])
    test_time_dfs = manager.list([])

    t = torch.cuda.get_device_properties(0).total_memory

    for hidden_dim in hidden_dim_to_try:
        for learning_rate in learning_rate_to_try:
            print(hidden_dim, learning_rate)
            torch.cuda.empty_cache()
            gc.collect()

            #  just based on amount of stuff that will fit on an 11 GB GPU
            if hidden_dim <= 1400:
                max_num_processes = 5
            elif hidden_dim <= 3000:
                max_num_processes = 3
            elif hidden_dim <= 4500:
                max_num_processes = 2
            elif hidden_dim <= 8000:
                max_num_processes = 1
            else:
                raise ValueError("Hidden dimension is too large :'(")

            pool = mp.Pool(processes=max_num_processes)

            attn_bool_vectors = [[True, False, False, False],
                                 [False, True, False, False],
                                 [False, False, True, False],
                                 [False, False, False, True],
                                 [False, False, False, False]]
            # if max_num_processes > 0:
            input_tuples = [(0,
                             hidden_dim,
                             learning_rate,
                             exp33_hp['depth'],
                             attn_bool_vector,
                             results_subdirectory,
                             train_time_dfs,
                             test_time_dfs,
                             True) for attn_bool_vector in attn_bool_vectors]
            pool.starmap(run_exp33_on_conditions, input_tuples)
            pool.close()

    agg_train_time_df = pd.concat(train_time_dfs)
    agg_test_time_df = pd.concat(test_time_dfs)
    agg_train_time_df.to_csv(join(results_subdirectory, 'train_time.csv'))
    agg_test_time_df.to_csv(join(results_subdirectory, 'test_time.csv'))

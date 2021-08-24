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
import random
import math


cifar_data = get_cifar_data(
    data_dir=path_configs['data_dir'], batch_size=exp32_hp['batch_size'])


def run_exp32_on_conditions(run, hidden_dim, depth, attention_layers_as_bool, results_dir,
                            train_time_list, test_time_list, sporadic_list):
    # set up tensorboard

    if not any(attention_layers_as_bool):
        architecture = 'Fully Feedforward'
    else:
        d = {0: '0th',
             1: '1st',
             2: '2nd',
             3: '3rd'}
        architecture = 'Attention in the {layer_ord} Layer'.format(
            layer_ord=d[attention_layers_as_bool.index(True)])

    unique_run_name = architecture + ',' + str(run)
    tensorboard_run_path = join(results_dir, unique_run_name, str(time.time()))
    writer = SummaryWriter(tensorboard_run_path)

    # set up data, model, diligence calculations
    model = ConstantWidthDeepNet(input_dim=cifar_data['x_size'][1],
                                 hidden_dim=hidden_dim,
                                 depth=depth,
                                 output_dim=len(cifar_data['classes']),
                                 with_attention=attention_layers_as_bool)

    initial_weights = [move(torch.clone(model.fetch_value_weights(i)))
                       for i in range(model.depth)]
    initial_norms = [torch.linalg.norm(w) for w in initial_weights]
    model = move(model)

    # set up training
    loss_fn = exp32_hp['base_loss_fn']()
    optimizer = exp32_hp['optimizer'](
        model.parameters(), lr=exp32_hp['learning_rate'])
    train_time_records = []
    test_time_records = []
    prev_intralayer_outputs = []
    next_activation_batch_idx = 1
    activation_batch_rate = 1.2
    cumul_grad_per_neuron = [move(torch.zeros(layer.V.weight.size()))
                             for layer in model._modules['layers']]
    sporadic_records = []

    # Get a sample of neurons in each layer
    neuron_layer_to_idxs = {layer_idx: random.sample(
        range(hidden_dim), exp32_hp['num_neurons_to_track']) for layer_idx in range(4)}

    for epoch in range(exp32_hp['num_epochs']):
        gc.collect()
        test_time_record = {'epoch': epoch,
                            'architecture': architecture,
                            'run': run}

        # train loop
        model.train()
        for (batch_num, (x, y)) in enumerate(cifar_data['train_loader']):
            # ORDINARY TRAIN LOOP CALCULATIONS

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
                                 'run': run}

            writer.add_scalar('Train Loss', loss, step_idx)
            train_time_record['train_loss'] = loss.item()

            train_time_records.append(train_time_record)

            # GET CONSISTENT TRAIN-TIME METRICS TODO
            # 1. Intralayer Norms
            # 2. Activation Correlations
            # 3. Save the subset of neuron-wise activations, gradients for time series.  Computer autocorrelation post facto.
            # 4. Hoyer sparsity measure: https://arxiv.org/pdf/0811.4706v2.pdf, https://math.stackexchange.com/questions/117860/how-to-define-sparseness-of-a-vector

            if prev_intralayer_outputs:
                for layer_idx, intralayer_outputs in enumerate(intralayer_outputs_by_layer):
                    # 1. Intralayer Norms
                    if 'q_norm' in intralayer_outputs:  # TODO: fix check to be not JANK
                        writer.add_scalar("Layer {} q_hadamard_k norm".format(
                            layer_idx), intralayer_outputs['q_hadamard_k_norm'], step_idx)
                        train_time_record['Layer {} q_hadamard_k norm'] = intralayer_outputs['q_hadamard_k_norm']

                    # 2. Activation Correlations
                    X = intralayer_outputs['output']
                    Y = prev_intralayer_outputs[layer_idx]['output']
                    N = torch.numel(X)

                    # calculate correlation with previous timestep's activations
                    # X := current activations, BxD
                    # Y := prev activations, BxD
                    X_bar = torch.mean(X)
                    Y_bar = torch.mean(Y)

                    X_res = X - X_bar
                    Y_res = Y - Y_bar

                    X_mse = torch.square(X_res)
                    Y_mse = torch.square(Y_res)

                    X_std = torch.sqrt(1 / N * torch.sum(X_mse))
                    Y_std = torch.sqrt(1 / N * torch.sum(Y_mse))

                    cov = torch.mean(X_res * Y_res)
                    corr = cov / (X_std * Y_std)

                    train_time_record['layer_{}_correlation'.format(
                        layer_idx)] = corr.item()
                    writer.add_scalar("Layer {} activation correlation".format(
                        layer_idx), corr, step_idx)

                    # 3. Save subset of neurons' activations and gradients
                    activation_slice = intralayer_outputs['v'][:, neuron_layer_to_idxs[layer_idx]].flatten(
                    ).tolist()  # takes B x hidden_dim -> B x sample_size
                    assert len(
                        activation_slice) == exp32_hp['batch_size'] * exp32_hp['num_neurons_to_track']
                    train_time_record['layer_{}_neuron_level_activations'.format(
                        layer_idx)] = activation_slice

                    gradient_xsection = model.fetch_value_weights(
                        layer_idx).grad  # B x layer_(i+1)_width x layer_(i)_width
                    breakpoint()
                    # B x sample_size x layer_(i)_width
                    gradient_xsection = gradient_xsection[:,
                                                          :, neuron_layer_to_idxs[layer_idx]]
                    gradient_norm_slice = torch.sum(torch.square(gradient_xsection), dim=[
                                                    0, 2]).squeeze().flatten().tolist()
                    train_time_record['layer_{}_neuron_level_gradient_norms'.format(
                        layer_idx)] = gradient_norm_slice

                    # 4. Hoyer sparsity measure
                    l1_X = torch.linalg.norm(X, ord=1)
                    l2_X = torch.linalg.norm(X, ord=2)
                    hoyer_sparsity = float(
                        (math.sqrt(N) - l1_X / l2_X) / (math.sqrt(N) - 1))
                    train_time_record['layer_{}_hoyer_sparsity'.format(
                        layer_idx)] = hoyer_sparsity
                    writer.add_scalar("Layer {} Hoyer sparsity".format(
                        layer_idx), hoyer_sparsity, step_idx)

                    del Y, Y_std, Y_res, Y_mse, Y_bar, X_std, X_res, X_mse, X_bar, corr, cov, N, gradient_xsection, l1_X, l2_X

            # GET SPORADIC METRICS TODO
            # 1. Activation Distribution
            # 2. Gini Measure on Activations
            # 3. Gradient Distribution
            # 4. Norm, Stable Rank of Layer Weight Matrix

            if step_idx >= int(next_activation_batch_idx):
                sporadic_record = {'batch_idx': step_idx,
                                   'architecture': architecture,
                                   'run': run}

                # 1. Activation Distribution (can't add to record bc too big)
                writer.add_histogram(
                    "Layer {} activation distribution".format(layer_idx), X, step_idx)

                # 2. Gini Measure on Activations
                gini_coeff = batched_gini(X)
                sporadic_record['layer_{}_gini'.format(layer_idx)] = gini_coeff
                writer.add_scalar("Layer {} Gini".format(
                    layer_idx), gini_coeff, step_idx)

                # 3. Gradient Distribution. and
                # 4. Norm, Stable Rank of Layer Weight Matrix
                quantile_dict = {}
                cum_quantile_dict = {}
                V_nuc_norm_dict = {}
                V_stable_rank_dict = {}
                for i, layer in enumerate(model._modules['layers']):
                    quantiles = np.quantile(
                        layer.V.weight.grad.flatten().tolist(), exp3_hp['quantiles'])
                    quantile_dict[i] = quantiles
                    writer.add_scalar("Layer {} 20th percentile gradient".format(i), quantiles[2],
                                      step_idx)
                    writer.add_scalar("Layer {} 50th percentile gradient".format(i), quantiles[5],
                                      step_idx)
                    writer.add_scalar("Layer {} 80th percentile gradient".format(i), quantiles[8],
                                      step_idx)

                    cumul_grad_per_neuron[i] = (layer.V.weight.grad + cumul_grad_per_neuron[i] * step_idx) / (
                        step_idx + 1)
                    cum_quantiles = np.quantile(cumul_grad_per_neuron[i].flatten().tolist(),
                                                exp32_hp['quantiles'])
                    cum_quantile_dict[i] = cum_quantiles
                    writer.add_scalar("Layer {} 20th percentile gradient average".format(i), cum_quantiles[2],
                                      step_idx)
                    writer.add_scalar("Layer {} 50th percentile gradient average".format(i), cum_quantiles[5],
                                      step_idx)
                    writer.add_scalar("Layer {} 80th percentile gradient average".format(i), cum_quantiles[8],
                                      step_idx)

                    V_nuc_norm = float(torch.linalg.matrix_norm(
                        layer.V.weight, ord='nuc'))
                    writer.add_scalar("Layer {} weight nuclear norm".format(
                        {i}), V_nuc_norm, step_idx)
                    V_nuc_norm_dict[i] = V_nuc_norm

                    V_stable_rank = float(torch.linalg.matrix_norm(
                        layer.V.weight, ord='fro') / torch.linalg.matrix_norm(layer.V.weight, ord=2))
                    writer.add_scalar("Layer {} weight stable rank".format(
                        {i}), V_stable_rank, step_idx)
                    V_stable_rank_dict[i] = V_stable_rank

                sporadic_record['gradient_quantiles'] = quantile_dict
                sporadic_record['cumulative_gradient_quantiles'] = cum_quantile_dict
                sporadic_record['weight_nuclear_norms'] = V_nuc_norm
                sporadic_record['weight_stable_rank'] = V_stable_rank_dict

                sporadic_records.append(sporadic_record)

                next_activation_batch_idx = max(
                    step_idx, activation_batch_rate * next_activation_batch_idx)

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
            dist = torch.linalg.norm(model.fetch_value_weights(
                i) - initial_weights[i]) / initial_norms[i]
            writer.add_scalar('Diligence of Layer ' + str(i), dist, epoch)
            test_time_record['layer_{}_diligence'.format(i)] = dist.item()

            del dist

        test_time_records.append(test_time_record)

    train_time_df = pd.DataFrame(train_time_records)
    test_time_df = pd.DataFrame(test_time_records)
    sporadic_df = pd.DataFrame(sporadic_records)

    train_time_list.append(train_time_df)
    test_time_list.append(test_time_df)
    sporadic_list.append(sporadic_df)


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
    sporadic_dfs = manager.list([])

    pool = mp.Pool(processes=5)

    attn_bool_vectors = [[True, False, False, False],
                         [False, True, False, False],
                         [False, False, True, False],
                         [False, False, False, True],
                         [False, False, False, False]]
    # if max_num_processes > 0:
    input_tuples = [(0,
                     exp32_hp['hidden_dim'],
                     exp32_hp['depth'],
                     attn_bool_vector,
                     results_subdirectory,
                     train_time_dfs,
                     test_time_dfs,
                     sporadic_dfs) for attn_bool_vector in attn_bool_vectors]
    pool.starmap(run_exp32_on_conditions, input_tuples)
    pool.close()

    agg_train_time_df = pd.concat(train_time_dfs)
    agg_test_time_df = pd.concat(test_time_dfs)
    agg_sporadic_df = pd.concat(sporadic_dfs)
    agg_train_time_df.to_csv(join(results_subdirectory, 'train_time.csv'))
    agg_test_time_df.to_csv(join(results_subdirectory, 'test_time.csv'))
    agg_sporadic_df.to_csv(join(results_subdirectory, 'sporadic.csv'))

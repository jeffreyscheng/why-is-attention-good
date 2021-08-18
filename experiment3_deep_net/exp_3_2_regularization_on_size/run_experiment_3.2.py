from utils.data_stuff import get_cifar_data
from utils.model_architectures import ConstantWidthDeepNet
from utils.evaluation import batched_gini
from params import machine_configs, path_configs, exp3_hp, move
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import time
import torch
import pandas as pd
import numpy as np
import itertools
import gc


def run_exp32_on_conditions(name, hidden_dim, depth, attn_bool_vector):
    run_path = join(path_configs['results_dir'], name, str(time.time()))
    writer = SummaryWriter(run_path)
    cifar_data = get_cifar_data(data_dir=path_configs['data_dir'], batch_size=exp3_hp['batch_size'])
    model = ConstantWidthDeepNet(input_dim=cifar_data['x_size'][1],
                                 hidden_dim=hidden_dim,
                                 depth=depth,
                                 output_dim=len(cifar_data['classes']),
                                 with_attention=attn_bool_vector)

    initial_weights = [move(torch.clone(model.fetch_value_weights(i))) for i in range(model.depth)]
    initial_norms = [torch.linalg.norm(w) for w in initial_weights]

    model = move(model)
    loss_fn = exp3_hp['base_loss_fn']()
    optimizer = exp3_hp['optimizer'](model.parameters(), lr=exp3_hp['learning_rate'])
    prev_intralayer_outputs = []

    cumul_grad_per_neuron = [move(torch.zeros(layer.V.weight.size())) for layer in model._modules['layers']]

    train_time_records = []
    test_time_records = []
    sporadic_records = pd.DataFrame(columns=['batch_idx', 'architecture', 'run', 'layer', 'activations'])
    sporadic_records.to_csv(join(run_path, 'sporadic.csv'), index=False)

    next_activation_batch_idx = 1
    activation_batch_rate = 1.2

    for epoch in range(exp3_hp['num_epochs']):
        gc.collect()
        model.train()

        test_time_record = {'epoch': epoch,
                            'architecture': name,
                            'run': 0}  # TODO

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
                                 'architecture': name,
                                 'run': 0}  # TODO
            # sporadic_record = train_time_record.copy()

            if prev_intralayer_outputs:
                for layer_idx, intralayer_outputs in enumerate(intralayer_outputs_by_layer):
                    # if 'q_norm' in intralayer_outputs:  # TODO: fix check to be not JANK
                    #     writer.add_scalar("Layer {} q norm".format(layer_idx), intralayer_outputs['q_norm'], step_idx)
                    #     writer.add_scalar("Layer {} k norm".format(layer_idx), intralayer_outputs['k_norm'], step_idx)
                    #     writer.add_scalar("Layer {} v norm".format(layer_idx), intralayer_outputs['v_norm'], step_idx)
                    #     writer.add_scalar("Layer {} q_hadamard_k norm".format(layer_idx), intralayer_outputs['q_hadamard_k_norm'], step_idx)
                    #     writer.add_scalar("Layer {} attn weight norm".format(layer_idx), intralayer_outputs['attn_weight_norm'], step_idx)
                    #     writer.add_scalar("Layer {} output norm".format(layer_idx), intralayer_outputs['output_norm'], step_idx)
                    #
                    #     writer.add_scalar("Layer {} q mean".format(layer_idx), intralayer_outputs['q_mean'], step_idx)
                    #     writer.add_scalar("Layer {} k mean".format(layer_idx), intralayer_outputs['k_mean'], step_idx)
                    #     writer.add_scalar("Layer {} v mean".format(layer_idx), intralayer_outputs['v_mean'], step_idx)
                    #     writer.add_scalar("Layer {} q_hadamard_k mean".format(layer_idx),
                    #                       intralayer_outputs['q_hadamard_k_mean'], step_idx)
                    #     writer.add_scalar("Layer {} attn weight mean".format(layer_idx),
                    #                       intralayer_outputs['attn_weight_mean'], step_idx)
                    #     writer.add_scalar("Layer {} output mean".format(layer_idx), intralayer_outputs['output_mean'],
                    #                       step_idx)

                    X = intralayer_outputs['output']
                    Y = prev_intralayer_outputs[layer_idx]['output']

                    # calculate correlation with previous timestep's activations
                    # X := current activations, BxD
                    # Y := prev activations, BxD
                    X_bar = torch.mean(X)
                    Y_bar = torch.mean(Y)

                    X_res = X - X_bar
                    Y_res = Y - Y_bar

                    X_mse = torch.square(X_res)
                    Y_mse = torch.square(Y_res)

                    X_std = torch.sqrt(1 / torch.numel(X) * torch.sum(X_mse))
                    Y_std = torch.sqrt(1 / torch.numel(X) * torch.sum(Y_mse))

                    cov = torch.mean(X_res * Y_res)
                    corr = cov / (X_std * Y_std)

                    train_time_record['layer_{}_correlation'.format(layer_idx)] = corr.item()
                    writer.add_scalar("Layer {} activation correlation".format(layer_idx), corr, step_idx)

                    if step_idx >= int(next_activation_batch_idx):
                        writer.add_histogram("Layer {} activation distribution".format(layer_idx), X, step_idx)
                        gini_coeff = batched_gini(X)
                        train_time_record['layer_{}_gini'.format(layer_idx)] = gini_coeff
                        writer.add_scalar("Layer {} Gini".format(layer_idx), gini_coeff, step_idx)

                        quantile_dict = {}
                        cum_quantile_dict = {}
                        V_norm_dict = {}
                        for i, layer in enumerate(model._modules['layers']):
                            quantiles = np.quantile(layer.V.weight.grad.flatten().tolist(), exp3_hp['quantiles'])
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
                                                        exp3_hp['quantiles'])
                            cum_quantile_dict[i] = cum_quantiles
                            writer.add_scalar("Layer {} 20th percentile gradient average".format(i), cum_quantiles[2],
                                              step_idx)
                            writer.add_scalar("Layer {} 50th percentile gradient average".format(i), cum_quantiles[5],
                                              step_idx)
                            writer.add_scalar("Layer {} 80th percentile gradient average".format(i), cum_quantiles[8],
                                              step_idx)

                            V_norm = torch.linalg.norm(layer.V.weight, ord='nuc')
                            writer.add_scalar("Layer {} weight norm".format({i}), V_norm, step_idx)
                            V_norm_dict[i] = V_norm

                        sporadic_df = pd.read_csv(join(run_path, 'sporadic.csv'))
                        sporadic_df = sporadic_df.append({'batch_idx': step_idx,
                                                          'architecture': name,
                                                          'run': 0,
                                                          'layer': layer_idx,
                                                          'activations': X.cpu().detach().numpy(),
                                                          'gradient_quantiles': quantile_dict,
                                                          'v_norm': V_norm_dict},
                                                         ignore_index=True)
                        sporadic_df.to_csv(join(run_path, 'sporadic.csv'), index=False)
                        del sporadic_df
                        gc.collect()

                    del Y, Y_std, Y_res, Y_mse, Y_bar, X_std, X_res, X_mse, X_bar, corr, cov

                if step_idx >= int(next_activation_batch_idx):
                    next_activation_batch_idx = max(step_idx, activation_batch_rate * next_activation_batch_idx)

            prev_intralayer_outputs = intralayer_outputs_by_layer

            writer.add_scalar('Train Loss', loss, step_idx)
            train_time_record['train_loss'] = loss.item()

            train_time_records.append(train_time_record)
            # sporadic_records.append(sporadic_record)

            del x, y, y_hat, loss

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

        test_loss = (sum(losses) / len(losses))
        test_accuracy = (sum(accuracy) / len(accuracy))
        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Test Accuracy', test_accuracy, epoch)
        test_time_record['test_loss'] = test_loss.item()
        test_time_record['test_accuracy'] = test_accuracy

        for i in range(model.depth):
            dist = torch.linalg.norm(model.fetch_value_weights(i) - initial_weights[i]) / initial_norms[i]
            writer.add_scalar('Diligence of Layer ' + str(i), dist, epoch)
            test_time_record['layer_{}_diligence'.format(i)] = dist.item()

        test_time_records.append(test_time_record)

    train_time_df = pd.DataFrame(train_time_records)
    test_time_df = pd.DataFrame(test_time_records)
    # sporadic_df = pd.DataFrame(sporadic_records)

    train_time_df.to_csv(join(run_path, 'train_time.csv'))
    test_time_df.to_csv(join(run_path, 'test_time.csv'))
    # sporadic_df.to_pickle(join(run_path, 'sporadic.pkl'))


for hidden_dim in [1000]:
    for depth in [4]:
        for attn_bool_vector in [[False, False, False, False],
                                 [True, False, False, False],
                                 [False, True, False, False],
                                 [False, False, True, False],
                                 [False, False, False, True]]:
            # for attn_bool_vector in [x for x in itertools.product([False, True], repeat=depth)]:
            #     if any(attn_bool_vector):
            run_exp32_on_conditions(
                'CWDN{hd:d},{dep:d},{attn}'.format(hd=hidden_dim, dep=depth, attn=str(attn_bool_vector)), hidden_dim,
                depth, attn_bool_vector)

from utils.data_stuff import get_cifar_data
from utils.model_architectures import ConstantWidthDeepNet
from params import machine_configs, path_configs, exp3_hp, move
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import time
import torch
import itertools


def run_exp3_on_conditions(name, hidden_dim, depth, attn_bool_vector):
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

    for epoch in range(exp3_hp['num_epochs']):
        model.train()
        for (batch_num, (x, y)) in enumerate(cifar_data['train_loader']):
            x = move(x)
            y = move(y)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train Loss', loss, epoch * len(cifar_data['train_loader']) + batch_num)

        model.eval()
        losses = []
        accuracy = []
        for (batch_num, (x, y)) in enumerate(cifar_data['test_loader']):
            x = move(x)
            y = move(y)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            losses.append(loss)
            accuracy += (torch.argmax(y_hat, dim=1) == y).int().tolist()

        writer.add_scalar('Test Loss', sum(losses) / len(losses), epoch)
        writer.add_scalar('Test Accuracy', sum(accuracy) / len(accuracy), epoch)

        for i in range(model.depth):
            dist = torch.linalg.norm(model.fetch_value_weights(i) - initial_weights[i]) / initial_norms[i]
            writer.add_scalar('Diligence of Layer ' + str(i), dist, epoch)


# for hidden_dim in [100, 200, 500, 1000, 2000, 5000, 10000]:3
for hidden_dim in [5000, 1000]:
    for depth in [4]:
        for attn_bool_vector in [x for x in itertools.product([False, True], repeat=depth)]:
            if any(attn_bool_vector):
                run_exp3_on_conditions(
                    'CWDN{hd:d},{dep:d},{attn}'.format(hd=hidden_dim, dep=depth, attn=str(attn_bool_vector)), hidden_dim,
                    depth, attn_bool_vector)

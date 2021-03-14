from utils.data_stuff import get_cifar_data
from utils.model_architectures import LogisticRegression
from params import machine_configs, path_configs, exp1_hp, move
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import time
import torch


def run_exp1_on_conditions(name, with_attention=False):
    run_path = join(path_configs['results_dir'], name, str(time.time()))
    writer = SummaryWriter(run_path)
    cifar_data = get_cifar_data(data_dir=path_configs['data_dir'], batch_size=exp1_hp['batch_size'])
    model = LogisticRegression(input_dim=cifar_data['x_size'][1],
                               output_dim=len(cifar_data['classes']),
                               with_attention=with_attention)
    model = move(model)
    loss_fn = exp1_hp['base_loss_fn']()
    optimizer = exp1_hp['optimizer'](model.parameters(), lr=exp1_hp['learning_rate'])

    for epoch in range(exp1_hp['num_epochs']):
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


run_exp1_on_conditions('logistic_regr_without_attn', False)
# run_exp1_on_conditions('logistic_regr_with_attn', True)

from utils.data_stuff import get_cifar_data
from utils.model_architectures import MLP
from params import machine_configs, path_configs, exp2_hp, move
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import time
import torch


def run_exp2_on_conditions(name, hidden_dim, with_attention1=False, with_attention2=False):
    run_path = join(path_configs['results_dir'], name, str(time.time()))
    writer = SummaryWriter(run_path)
    cifar_data = get_cifar_data(data_dir=path_configs['data_dir'], batch_size=exp2_hp['batch_size'])
    model = MLP(input_dim=cifar_data['x_size'][1],
                hidden_dim=hidden_dim,
                output_dim=len(cifar_data['classes']),
                with_attention1=with_attention1,
                with_attention2=with_attention2)

    initial_fc1 = move(torch.clone(model.fc1.weight))
    initial_fc2 = move(torch.clone(model.fc2.weight))
    initial_fc1_norm = torch.linalg.norm(initial_fc1)
    initial_fc2_norm = torch.linalg.norm(initial_fc2)

    model = move(model)
    loss_fn = exp2_hp['base_loss_fn']()
    optimizer = exp2_hp['optimizer'](model.parameters(), lr=exp2_hp['learning_rate'])

    for epoch in range(exp2_hp['num_epochs']):
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

        dist_1 = torch.linalg.norm(model.fc1.weight - initial_fc1) / initial_fc1_norm
        dist_2 = torch.linalg.norm(model.fc2.weight - initial_fc2) / initial_fc2_norm

        writer.add_scalar('Laziness of First Layer', dist_1, epoch)
        writer.add_scalar('Laziness of Second Layer', dist_2, epoch)


for hidden_dim in [500]:
    run_exp2_on_conditions('MLP{hd:d}_wo_wo'.format(hd=hidden_dim), hidden_dim, False, False)
    run_exp2_on_conditions('MLP{hd:d}_with_wo'.format(hd=hidden_dim), hidden_dim, True, False)
    run_exp2_on_conditions('MLP{hd:d}_wo_with'.format(hd=hidden_dim), hidden_dim, False, True)
    run_exp2_on_conditions('MLP{hd:d}_with_with'.format(hd=hidden_dim), hidden_dim, False, True)

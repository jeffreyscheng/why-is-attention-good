import torch
import torchvision
import torchvision.transforms as transforms


def get_cifar_data(data_dir, batch_size):
    x_size = torch.Size([batch_size, 3 * 32 * 32])
    y_size = torch.Size([batch_size])

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Lambda(lambda x: torch.flatten(x, start_dim=0)),
         transforms.Lambda(lambda x: x.squeeze())])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return {'train_loader': trainloader,
            'test_loader': testloader,
            'x_size': x_size,
            'y_size': y_size,
            'classes': classes
            }

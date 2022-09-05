import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# import model architecture
from models.samplenet import SampleNet

# import stadle BasicClient class
from stadle import BasicClient

import argparse

def load_CIFAR100(batch=128, intensity=1.0, classes=None, sel_prob=1.0, def_prob=0.1):
    trainset_size = 60000

    # Set the dataset mask to perform training with biased data
    if (args.classes is not None):
        trainset = datasets.CIFAR100('./data',
                            train=True,
                            download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x * intensity)
                            ])
                          )
        classes = [int(c) for c in args.classes.split(',')]
        trainset.targets = torch.tensor(trainset.targets)
        mask = (trainset.targets == -1)
        for i in range(100):
            class_mask = (trainset.targets == i)
            mask_idx = class_mask.nonzero()
            class_size = len(mask_idx)
            size = sel_prob if (i in classes) else def_prob
            mask_idx = mask_idx[torch.randperm(class_size)][:int(class_size * size)]
            mask[mask_idx] = True

        trainset.data = trainset.data[mask]
        trainset.targets = trainset.targets[mask]
        trainset_size = len(trainset)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)

    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data',
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x * intensity)
                        ])),
            batch_size=batch,
            shuffle=True)
        train_loader = torch.tensor(train_loader)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x * intensity)
                       ])),
        batch_size=batch,
        shuffle=True)

    return {'train': train_loader, 'test': test_loader}, trainset_size


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='STADLE CIFAR10 Training')
    parser.add_argument('--agent_name', default='default_agent')
    parser.add_argument('--classes')
    parser.add_argument('--def_prob', type=float, default=0.1)
    parser.add_argument('--sel_prob', type=float, default=1.0)
    parser.add_argument('--reg_port')
    args = parser.parse_args()

    # Number of times of learning
    epoch = 150

    # For saving learning results
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': [],
    }

    # Build Network
    net: torch.nn.Module = SampleNet()
    loaders, trainset_size = load_CIFAR100(classes=args.classes, def_prob=args.def_prob, sel_prob=args.sel_prob)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
    client_config_path = r'config/config_agent.json'

    # Preload stadle_client
    stadle_client = BasicClient(config_file=client_config_path, agent_name=args.agent_name, reg_port=args.reg_port)
    stadle_client.set_bm_obj(net)

    for e in range(epoch):

        if (e % 2 == 0): # Set how many epochs the aggregation is executed
            # Don't send model at beginning of training
            if (e != 0):
                # Get model performance
                perf_dict = {
                            'performance':history['test_acc'][-1],
                            'accuracy' : history['test_acc'][-1],
                            'loss_training' : history['train_loss'][-1],
                            'loss_test' : history['test_loss'][-1]}
                # Send trained local model
                stadle_client.send_trained_model(net, perf_dict)

            # Recieve semi global model
            state_dict = stadle_client.wait_for_sg_model().state_dict()
            net.load_state_dict(state_dict)

        # Training
        loss = None
        net.train(True)

        for i, (data, target) in enumerate(loaders['train']):
            data = data.view(-1, 28 * 28)
            optimizer.zero_grad()
            output = net(data)
            loss = f.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('Training log: {} epoch ({} / {} train. data). Loss: {}'.format(e + 1, (i + 1) * 128,
                                                                                         trainset_size, loss.item()))
        history['train_loss'].append(loss.item())

        #Test
        net.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in loaders['test']:
                data = data.view(-1, 28 * 28)
                output = net(data)
                test_loss += f.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= 10000
        print('Test loss (avg): {}, Accuracy: {}'.format(test_loss,
                                                         correct / 10000))

        history['test_loss'].append(test_loss)
        history['test_acc'].append(correct / 10000)
import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
import sys

from torchvision.transforms import Compose
from tqdm import tqdm
from modeling import MLP_MNIST, MoE, CustomMNIST, Temperature_Scheduler, CustomCIFAR10, CNN_Cifar10_Large
import argparse

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for mini-batch training and evaluating. Default: 32')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Number of training epoch. Default: 10')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='Learning rate during optimization. Default: 1e-3')
parser.add_argument('--num_experts', type=int, default=10,
                    help='Number of Experts. Default: 10')
parser.add_argument('--topk', type=int, default=1,
                    help='Default: 1')
parser.add_argument('--data_set', choices=['Imagenet', 'Cifar10', 'Cifar100', 'MNIST'],
                    default='MNIST')
parser.add_argument('--routing_method', choices=['Noisy_Topk', 'Topk', 'Hash', 'Reinforce', 'Anneal', 'BASE'],
                    default='Topk')
parser.add_argument('--h_temp', type=float, default=2)
parser.add_argument('--l_temp', type=float, default=0.5)
parser.add_argument('--expert_type', choices=['CNN', 'MLP'],
                    default='MLP')
parser.add_argument("--use_moe", action='store_true', default=False)
parser.add_argument("--test_model", action='store_true', default=False)
parser.add_argument("--tensorboard", action='store_true', default=False)
args = parser.parse_args()

if __name__ == '__main__':

    batch_size = args.batch_size
    epochs = args.num_epochs
    learning_rate = args.learning_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data_set == 'MNIST':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(0.5, 0.5)])
        trainset = CustomMNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        testset = CustomMNIST(root='./data', train=False, download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        train_datasize = len(trainset)
        test_datasize = len(testset)
        input_size = 28 * 28
        output_size = 10
        config = {
            'input_size': 28 * 28,
            'output_size': 10,
            'dropout_rate': 0.2,
        }
        print('Dataset is ' + args.data_set)
        print('Training set size: %d' % train_datasize)
        print('Test set size: %d' % test_datasize)
    elif args.data_set == 'Cifar10':
        train_transform = Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])
        test_transform = Compose([
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])
        dataset = CustomCIFAR10(root='./data', train=True, download=True, transform=train_transform)
        train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000])
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        testset = CustomCIFAR10(root='./data', train=False, download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        train_datasize = len(train_set)
        val_datasize = len(val_set)
        test_datasize = len(testset)
        input_size = 3 * 32 * 32
        output_size = 10
        config = {
            'input_size': input_size,
            'output_size': output_size,
            'dropout_rate': 0.2,
        }
        print('Dataset is ' + args.data_set)
        print('Training set size: %d' % train_datasize)
        print('Validation set size: %d' % val_datasize)
        print('Test set size: %d' % test_datasize)
    else:
        raise Exception("Unsupported Dataset")

    if args.use_moe:
        model = MoE(input_size=config['input_size'], output_size=config['output_size'], num_experts=args.num_experts,
                    expert_type=args.expert_type, k=args.topk, routing_method=args.routing_method,
                    dataset=args.data_set, train_datasize=train_datasize, test_datasize=test_datasize).to(device)
    else:
        if args.expert_type == "MLP":
            model = MLP_MNIST().to(device)
        elif args.expert_type == "CNN":
            model = CNN_Cifar10_Large().to(device)
        else:
            raise Exception("Unsupported model type")
    print(model)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable Parameters: " + str(count_parameters(model)))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    start = time.perf_counter()

    if args.routing_method == 'Anneal':
        temp_sche = Temperature_Scheduler(args.l_temp, args.h_temp, epochs)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        correct = 0
        total = 0
        for i, data in enumerate(tqdm(trainloader, desc="Train Progress", file=sys.stdout)):
            # get the inputs; data is a list of [inputs, labels]
            index, inputs, labels = data
            index, inputs, labels = index.to(device), inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            if args.use_moe and args.routing_method in ["Noisy_Topk", "Topk"]:
                outputs, aux_loss = model(inputs, index)
            elif args.use_moe and args.routing_method == "Anneal":
                temp = temp_sche.get_temp(epoch)
                outputs, aux_loss = model(inputs, index, temp)
            elif args.use_moe and args.routing_method in ["Hash", "BASE"]:
                outputs = model(inputs, index)
                aux_loss = torch.tensor(0)
            else:
                outputs = model(inputs)
                aux_loss = torch.tensor(0)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss = loss + aux_loss
            total_loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()

        train_loss = running_loss / total * batch_size
        loss_train.append(train_loss)
        print('Epoch: ' + str(epoch + 1) + ' Training loss: %.4f' % (train_loss))
        train_acc = 100 * correct / total
        acc_train.append(train_acc)
        print('Epoch: ' + str(epoch + 1) + ' Training acc: %.2f' % (100 * correct / total))
        if args.tensorboard:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(valloader, desc="Val Progress", file=sys.stdout)):
                # get the inputs; data is a list of [inputs, labels]
                index, inputs, labels = data
                index, inputs, labels = index.to(device), inputs.to(device), labels.to(device)

                if args.use_moe and args.routing_method in ["Noisy_Topk", "Topk"]:
                    outputs, aux_loss = model(inputs, index)
                elif args.use_moe and args.routing_method == "Anneal":
                    temp = temp_sche.get_temp(epoch)
                    outputs, aux_loss = model(inputs, index, temp)
                elif args.use_moe and args.routing_method in ["Hash", "BASE"]:
                    outputs = model(inputs, index)
                    aux_loss = torch.tensor(0)
                else:
                    outputs = model(inputs)
                    aux_loss = torch.tensor(0)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()

        val_loss = running_loss / total * batch_size
        loss_val.append(val_loss)
        print('Epoch: ' + str(epoch + 1) + ' Val loss: %.4f' % (val_loss))
        val_acc = 100 * correct / total
        acc_val.append(val_acc)
        print('Epoch: ' + str(epoch + 1) + ' Val acc: %.2f' % (100 * correct / total))
        if args.tensorboard:
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Acc/val", val_acc, epoch)

    if args.test_model:
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(testloader, desc="Test Progress", file=sys.stdout)):
                # get the inputs; data is a list of [inputs, labels]
                index, inputs, labels = data
                index, inputs, labels = index.to(device), inputs.to(device), labels.to(device)

                if args.use_moe and args.routing_method in ["Noisy_Topk", "Topk"]:
                    outputs, aux_loss = model(inputs, index)
                elif args.use_moe and args.routing_method == "Anneal":
                    temp = temp_sche.get_temp(epoch)
                    outputs, aux_loss = model(inputs, index, temp)
                elif args.use_moe and args.routing_method in ["Hash", "BASE"]:
                    outputs = model(inputs, index)
                    aux_loss = torch.tensor(0)
                else:
                    outputs = model(inputs)
                    aux_loss = torch.tensor(0)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()

        test_loss = running_loss / total * batch_size
        print('Epoch: ' + str(epoch + 1) + ' Test loss: %.4f' % (test_loss))
        test_acc = 100 * correct / total
        print('Epoch: ' + str(epoch + 1) + ' Test acc: %.2f' % (100 * correct / total))

    end = time.perf_counter()
    print("运行时间为", round(end - start), 'seconds')
    if args.tensorboard:
        writer.flush()
        writer.close()
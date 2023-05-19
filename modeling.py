import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.distributions import Normal
from PIL import Image


class MLP_MNIST(nn.Module):
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, 512)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(512, 512)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(512, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.droput(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.droput(x)
        # add output layer
        x = self.fc3(x)
        return x



class CNN_Cifar10_Small(nn.Module):
    def __init__(self, drop_rate=0.3):
        super(CNN_Cifar10_Small, self).__init__()
        kernel_num_1 = 32
        kernel_num_2 = 64
        # Define your layers here
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=kernel_num_1, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(kernel_num_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(drop_rate)
        self.mp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=kernel_num_1, out_channels=kernel_num_2, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(kernel_num_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(drop_rate)
        self.mp2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(4096, 10)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # the 10-class prediction output is named as "logits"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if x.shape[0] == 0:
            logits = torch.empty(0, 10).to(device)
        else:
            output = self.conv1(x)
            output = self.bn1(output)
            output = self.relu1(output)
            output = self.dropout1(output)
            output = self.mp1(output)
            output = self.conv2(output)
            output = self.bn2(output)
            output = self.relu2(output)
            output = self.dropout2(output)
            output = self.mp2(output)
            output = output.view(output.size(0), -1)
            output = self.fc(output)
            logits = output

        return logits

class CNN_Cifar10_Large(nn.Module):
    def __init__(self, drop_rate=0.4):
        super(CNN_Cifar10_Large, self).__init__()
        kernel_num_1 = 128
        kernel_num_2 = 256
        # Define your layers here
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=kernel_num_1, kernel_size=(5, 5), padding=2)
        self.bn1 = nn.BatchNorm2d(kernel_num_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(drop_rate)
        self.mp1 = nn.MaxPool2d(5,  stride=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=kernel_num_1, out_channels=kernel_num_2, kernel_size=(7, 7), padding=3)
        self.bn2 = nn.BatchNorm2d(kernel_num_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(drop_rate)
        self.mp2 = nn.MaxPool2d(5, stride=4, padding=2)
        self.fc = nn.Linear(2304, 10)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # the 10-class prediction output is named as "logits"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if x.shape[0] == 0:
            logits = torch.empty(0, 10).to(device)
        else:
            output = self.conv1(x)
            output = self.bn1(output)
            output = self.relu1(output)
            output = self.dropout1(output)
            output = self.mp1(output)
            output = self.conv2(output)
            output = self.bn2(output)
            output = self.relu2(output)
            output = self.dropout2(output)
            output = self.mp2(output)
            output = output.view(output.size(0), -1)
            output = self.fc(output)
            logits = output

        return logits


class CustomMNIST(torchvision.datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super(CustomMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target


class CustomCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CustomCIFAR10, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target


class Temperature_Scheduler(object):
    def __init__(self, l_temp, h_temp, total_epochs):
        self.h_temp = h_temp
        self.l_temp = l_temp
        self.total_epochs = total_epochs
        assert(h_temp > 0 and 0 < l_temp < h_temp)

    def get_temp(self, epoch):
        return self.h_temp + (self.l_temp - self.h_temp) * epoch / self.total_epochs


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
        pass
    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


def balanced_assignment(scores, max_iterations=100):
    scores = scores.transpose(0,1)
    device = scores.device
    num_workers, num_jobs = scores.size()
    jobs_per_worker = num_jobs // num_workers
    value = scores.clone()

    iterations = 0
    cost = scores.new_zeros(1, num_jobs)

    jobs_with_bids = torch.zeros(num_workers).bool().to(device)
    eps = 1e-4

    while not jobs_with_bids.all():

        top_values, top_index = torch.topk(value, k=jobs_per_worker + 1, dim=1)

        # Each worker bids the difference in value between a job and the k+1th job

        bid_increments = top_values[:, :-1] - top_values[:, -1:] + eps
        bid_increments = bid_increments.to(device)
        bids = torch.scatter(torch.zeros(num_workers, num_jobs).to(device), dim=1,
                             index=top_index[:, :-1], src=bid_increments)

        if 0 < iterations < max_iterations:
            # If a worker won a job on the previous round, put in a minimal bid to retain
            # the job only if no other workers bid this round.

            bids[top_bidders, jobs_with_bids] = eps

        # Find the highest bidding worker per job
        top_bids, top_bidders = bids.max(dim=0)
        jobs_with_bids = top_bids > 0
        top_bidders = top_bidders[jobs_with_bids]

        # Make popular items more expensive
        cost += top_bids
        value = scores - cost

        if iterations < max_iterations:
            # If a worker won a job, make sure it appears in its top-k on the next round
            value[top_bidders, jobs_with_bids] = torch.inf
        else:
            value[top_bidders, jobs_with_bids] = scores[top_bidders, jobs_with_bids]
        iterations += 1
    batch_index = top_index[:, :-1].reshape(1, -1).squeeze(0)
    part_size = tuple(jobs_per_worker for i in range(num_workers))
    expert_index = torch.ones(100,1).type(torch.int64).to(device)
    for i in range(10):
        expert_index[i*10:(i+1)*10,:] = expert_index[i*10:(i+1)*10,:]*i
    expert_index = expert_index.type(torch.int64)
    return batch_index, part_size, expert_index


class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, expert_type, routing_method, dataset,
                 train_datasize, test_datasize, k=1):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.routing_method = routing_method
        self.dataset = dataset
        self.k = k
        # instantiate experts

        if expert_type == 'CNN' and dataset == 'Cifar10':
            self.experts = nn.ModuleList([CNN_Cifar10_Small() for i in range(self.num_experts)])
        elif expert_type == 'MLP' and dataset == 'MNIST':
            self.experts = nn.ModuleList([MLP_MNIST() for i in range(self.num_experts)])
        else:
            raise Exception("Unsupported Expert Type")

        if self.routing_method=='Topk' or self.routing_method=='Noisy_Topk':
            self.w_gate = nn.Parameter(0.1 * torch.rand(input_size, num_experts), requires_grad=True)
            self.w_noise = nn.Parameter(0.1 * torch.rand(input_size, num_experts), requires_grad=True)
            self.register_buffer("mean", torch.tensor([0.0]))
            self.register_buffer("std", torch.tensor([1.0]))
            self.softplus = nn.Softplus()
            self.softmax = nn.Softmax(1)
        elif self.routing_method == 'Hash':
            self.train_hash_list = torch.randint(low=0, high=self.num_experts, size=(train_datasize,))
            self.test_hash_list = torch.randint(low=0, high=self.num_experts, size=(test_datasize,))
        elif self.routing_method == 'Anneal':
            self.w_gate = nn.Parameter(0.1 * torch.rand(input_size, num_experts), requires_grad=True)
            self.softmax = nn.Softmax(1)
        elif self.routing_method == 'BASE':
            expert_centroids = torch.empty(input_size, num_experts)
            torch.nn.init.orthogonal_(expert_centroids, gain=0.1)
            self.register_parameter(
                "expert_centroids", torch.nn.Parameter(expert_centroids)
            )
        else:
            raise Exception("Unsupported routing method")
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, temp=1, noise_epsilon=1e-2):
        x = x.view(x.shape[0], -1)
        clean_logits = x @ self.w_gate
        if self.routing_method == "Noisy_Topk" and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits/temp)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.routing_method == "Noisy_Topk" and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, index, temp=1, loss_coef=1e-2):
        if self.routing_method == "Noisy_Topk" or self.routing_method == "Topk":
            gates, load = self.noisy_top_k_gating(x, self.training)
            # calculate importance loss
            importance = gates.sum(0)
            #
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= loss_coef

            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            gates = dispatcher.expert_to_gates()
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
            y = dispatcher.combine(expert_outputs)
            output = (y, loss)
        elif self.routing_method == "Anneal":
            gates, load = self.noisy_top_k_gating(x, self.training, temp=temp)
            # calculate importance loss
            importance = gates.sum(0)
            #
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= loss_coef

            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            gates = dispatcher.expert_to_gates()
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
            y = dispatcher.combine(expert_outputs)
            output = (y, loss)
        elif self.routing_method=="Hash":
            if self.training:
                hashlist = self.train_hash_list.to(x.device)
            else:
                hashlist = self.test_hash_list.to(x.device)
            gate = hashlist[index.view(-1)]
            order = gate.argsort(0)
            num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0)
            x = x[order]
            x = x.split(num_tokens.tolist(), dim=0)
            y = [self.experts[i].forward(x[i]) for i in range(self.num_experts) for i in range(self.num_experts)]
            y = torch.vstack(y)
            y = y[order.argsort(0)]
            output = y
        elif self.routing_method == 'BASE':
            features = x.reshape(x.size(0), -1)
            token_expert_affinities = features.matmul(
                self.expert_centroids
            )
            if self.training:
                with torch.no_grad():
                    batch_index, part_size, expert_index = balanced_assignment(token_expert_affinities)

                inp_exp = x[batch_index].squeeze(1)
                expert_inputs = torch.split(inp_exp, part_size, dim=0)
                expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]

                stitched = torch.cat(expert_outputs, 0).exp()
                gates_exp = token_expert_affinities[batch_index.flatten()]
                _nonzero_gates = torch.gather(gates_exp, 1, expert_index)
                stitched = stitched.mul(torch.sigmoid(_nonzero_gates))
                zeros = torch.zeros(x.size(0), expert_outputs[-1].size(1), requires_grad=True, device=stitched.device)

                # combine samples that have been processed by the same k experts
                combined = zeros.index_add(0, batch_index, stitched.float())
                # add eps to all zero values in order to avoid nans when going back to log space
                combined[combined == 0] = np.finfo(float).eps
                # back to log space
                output = combined.log()
            else:
                logits = torch.sigmoid(token_expert_affinities)
                # calculate topk + 1 that will be needed for the noisy gates
                top_logits, top_indices = logits.topk(1, dim=1)
                zeros = torch.zeros_like(logits, requires_grad=True)
                gates = zeros.scatter(1, top_indices, top_logits)

                dispatcher = SparseDispatcher(self.num_experts, gates)
                expert_inputs = dispatcher.dispatch(x)
                expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
                output = dispatcher.combine(expert_outputs)
        else:
            raise Exception("Unsupported routing method")
        return output

import torch.nn as nn
import torch
import sys
from fairseq import utils
from fairseq.distributed import utils as distributed_utils
from fairseq.modules.layer_norm import LayerNorm


class BaseLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_workers = distributed_utils.get_data_parallel_world_size()
        expert_centroids = torch.empty(self.num_workers, args.decoder_embed_dim)
        torch.nn.init.orthogonal_(expert_centroids, gain=0.1)
        self.register_parameter(
            "expert_centroids", torch.nn.Parameter(expert_centroids)
        )
        self.expert_network = nn.Sequential(
            *([BaseSublayer(args) for _ in range(args.base_sublayers)])
        )
        self.expert_id = distributed_utils.get_data_parallel_rank()
        self.shuffle = args.base_shuffle
        self.cpp = self

        # Add a special attribute to the expert parameters, so we know not to sync their gradients
        for param in self.expert_network.parameters():
            param.expert = True

    def forward(self, input_features, *args, **kwargs):
        features = input_features.reshape(-1, input_features.size(-1))
        is_training = input_features.requires_grad

        if self.shuffle and is_training:
            # Send each token to a random worker, to break correlations within the batch
            shuffle_sort = torch.randperm(features.size(0), device=features.device)
            features = All2All.apply(features[shuffle_sort])

        with torch.no_grad():
            # Compute similarity of each token to each expert, for routing
            token_expert_affinities = features.matmul(
                self.expert_centroids.transpose(0, 1)
            )

        # Compute which token goes to which expert
        sort_by_expert, input_splits, output_splits = (
            self.balanced_assignment(token_expert_affinities)
            if is_training
            else self.greedy_assignment(token_expert_affinities)
        )
        # Swap these tokens for the right ones for our expert
        routed_features = All2All.apply(
            features[sort_by_expert], output_splits, input_splits
        )

        if routed_features.size(0) > 0:
            # Mix in the expert network based on how appropriate it is for these tokens
            alpha = torch.sigmoid(
                routed_features.mv(self.expert_centroids[self.expert_id])
            ).unsqueeze(1)
            routed_features = (
                alpha * self.expert_network(routed_features)
                + (1 - alpha) * routed_features
            )
        # Return to original worker and ordering
        result = All2All.apply(routed_features, input_splits, output_splits)[
            self.inverse_sort(sort_by_expert)
        ]

        if self.shuffle and is_training:
            # Undo shuffling
            result = All2All.apply(result)[self.inverse_sort(shuffle_sort)]

        # Return additional Nones for compatibility with TransformerDecoderLayer
        return result.view(input_features.size()), None, None


    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(
            0, order, torch.arange(0, order.size(0), device=order.device)
        )


    def balanced_assignment(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return self.cpp.balanced_assignment(scores), None, None


    # Assigns each token to the top k experts
    def greedy_assignment(self, scores, k=1):
        token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
        token_to_workers, sort_ordering = torch.sort(token_to_workers)
        worker2token = sort_ordering // k

        # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
        output_splits = torch.zeros(
            (self.num_workers,), dtype=torch.long, device=scores.device
        )
        workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
        output_splits[workers] = counts
        # Tell other workers how many tokens to expect from us
        input_splits = All2All.apply(output_splits)
        return worker2token, input_splits.tolist(), output_splits.tolist()


    def load_assignment(self):
        try:
            from fairseq import libbase

            return libbase

        except ImportError as e:
            sys.stderr.write(
                "ERROR: missing libbase. run `python setup.py build_ext --inplace`\n"
            )
            raise e



class BaseSublayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu") or "relu"
        )
        self.norm = LayerNorm(args.decoder_embed_dim, export=False)
        self.ff1 = torch.nn.Linear(args.decoder_embed_dim, args.decoder_ffn_embed_dim)
        self.ff2 = torch.nn.Linear(args.decoder_ffn_embed_dim, args.decoder_embed_dim)
        self.ff2.weight.data.zero_()

    def forward(self, xs):
        return xs + self.ff2(self.activation_fn(self.ff1(self.norm(xs))))


# Wraps torch.distributed.all_to_all_single as a function that supports autograd
class All2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, input_splits=None, output_splits=None):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits

        ys = (
            torch.empty_like(xs)
            if output_splits is None
            else xs.new_empty(size=[sum(output_splits)] + list(xs.size()[1:]))
        )
        torch.distributed.all_to_all_single(
            ys, xs, output_split_sizes=output_splits, input_split_sizes=input_splits
        )
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        result = (
            torch.empty_like(grad_output)
            if ctx.input_splits is None
            else grad_output.new_empty(
                size=[sum(ctx.input_splits)] + list(grad_output.size()[1:])
            )
        )
        torch.distributed.all_to_all_single(
            result,
            grad_output,
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
        )
        return result, None, None

def balanced_assignment(scores, max_iterations=100):
    num_workers, num_jobs = scores.size()
    jobs_per_worker = num_jobs // num_workers
    value = scores.clone()

    iterations = 0
    cost = scores.new_zeros(1, num_jobs)

    jobs_with_bids = torch.zeros(num_workers).bool()
    eps = 1e-4

    while not jobs_with_bids.all():

        top_values, top_index = torch.topk(value, k=jobs_per_worker + 1, dim=1)

        # Each worker bids the difference in value between a job and the k+1th job

        bid_increments = top_values[:, :-1] - top_values[:, -1:] + eps
        bids = torch.scatter(torch.zeros(num_workers, num_jobs), dim=1,
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
    print(iterations)
    return top_index[:, :-1]


def greedy_assignment(scores, k=1):
    token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
    token_to_workers, sort_ordering = torch.sort(token_to_workers)
    worker2token = sort_ordering // k

    # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
    output_splits = torch.zeros(
        (self.num_workers,), dtype=torch.long, device=scores.device
    )
    workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
    output_splits[workers] = counts
    # Tell other workers how many tokens to expect from us
    input_splits = All2All.apply(output_splits)
    return worker2token, input_splits.tolist(), output_splits.tolist()


if __name__ == "__main__":
    num_workers = 100
    decoder_embed_dim = 10
    input_features = torch.rand((10, 10))
    features = input_features.reshape(-1, input_features.size(-1))
    is_training = input_features.requires_grad

    expert_centroids = torch.empty(num_workers, decoder_embed_dim)
    torch.nn.init.orthogonal_(expert_centroids, gain=0.1)


    token_expert_affinities = features.matmul(
        expert_centroids.transpose(0, 1)
    )
    is_training = True
    # Compute which token goes to which expert
    sort_by_expert, input_splits, output_splits = (
        balanced_assignment(token_expert_affinities)
        if is_training
        else greedy_assignment(token_expert_affinities)
    )
    # Swap these tokens for the right ones for our expert
    print(token_expert_affinities)
    print(sort_by_expert)


    # if features.size(0) > 0:
    #     # Mix in the expert network based on how appropriate it is for these tokens
    #     alpha = torch.sigmoid(
    #         features.mv(expert_centroids[expert_id])
    #     ).unsqueeze(1)
    #     features = (
    #             alpha * expert_network(features)
    #             + (1 - alpha) * features
    #     )



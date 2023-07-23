import torch
import torch.distributed as dist
from tqdm import tqdm
import torch.nn.functional as F

from . import utils
from .utils import oprint, rprint

@torch.no_grad()
def accuracy(logits, targets):
    # logits are [B, seq_len, vocab_size],  targets are [B, seq_len] integers
    top_logits = torch.argmax(logits, dim=-1)  # [B, seq_len]
    return (top_logits == targets).float().mean()

# @torch.no_grad()
# def perplexity(logits, targets):  # literally just exp(avg_cross_entropy_per_tok) -> perplexity per token
#     # logits are [B, seq_len, vocab_size],  targets are [B, seq_len] integers
#     vocab_size = logits.shape[-1]
#     # batch_size, seq_len, vocab_size = logits.shape[-1]
#     # log_probs = torch.log(F.softmax(logits, dim=-1)) # [B, seq_len, vocab_size]
#     # return log_probs.view(-1, vocab_size)[torch.arange(batch_size*seq_len), targets].mean()  # average over B and seq_len
#     loss_values = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
#     return torch.exp(loss_values)


@torch.no_grad()
def evaluate(net, dsets, exp_config, all_metrics):
    net.eval()
    # {metric_name1: {split1: Tensor, split2: Tensor, ...}, metric_name2: {split1: Tensor, split2: Tensor, ...}, ...}
    # set up tensors for sychronization
    # TODO: maybe this should just be a [num_metrics, num_eval] tensor? 
    epoch_metrics = {k: {split: torch.zeros(exp_config.num_eval).to(net.device) 
                         for split in dsets} 
                     for k in all_metrics}
    
    oprint("starting evaluation at", utils.get_time(), "mem", torch.cuda.memory_summary())

    for split in dsets:
        # print("starting split", split, utils.get_rank(), utils.get_time())
        for i, sample in tqdm(enumerate(dsets[split].dataloader())):
            if i >= exp_config.num_eval:  # so that we don't iterate over the entire training set while estimating tr_loss
                break
            if i <= 1:
                rprint("started", torch.cuda.memory_summary(), i)
            if isinstance(sample, dict):  # handle commavq dataset
                sample = sample['xy'].transpose(0,1)
            if i <= 1:
                rprint("sample.transpose", torch.cuda.memory_summary(), i)
            # print("started iterating", split)
            # print(sample)
            x,y = [el.to(net.device, non_blocking=True) for el in sample]
            if i <= 1:
                rprint("xy.cuda", torch.cuda.memory_summary(), i)
            # print("transferred sample")
            logits, loss = net(x, y)  # match nanoGPT order of return
            if i <= 1:
                rprint("net(x,y)", torch.cuda.memory_summary(), i)
            # calculate supported metrics
            for metric_name in epoch_metrics:
                if metric_name == "loss":  # calculate the metrics that are requested
                    metric_result = loss
                elif metric_name == "perplexity":
                    metric_result = torch.exp(loss)
                elif metric_name == "accuracy":
                    metric_result = accuracy(logits, y)
                else:
                    raise ValueError(f"Metric name {metric_name} is not supported.")
                epoch_metrics[metric_name][split][i] = metric_result
            if i <= 1:
                rprint("after full eval iter", torch.cuda.memory_summary(), i)
        # print("on split", split, 'rank', utils.get_rank(), "has loss", epoch_metrics["loss"][split], utils.get_time())
        if exp_config.ddp:
            for metric_name in epoch_metrics.keys():  # do this since you shouldn't iterate over a thing you are modifying
                # print("rank", utils.get_rank(), "starting all_reduce on ", metric_name, epoch_metrics[metric_name][split])
                # print(epoch_metrics[metric_name][split])
                dist.all_reduce(epoch_metrics[metric_name][split], op=dist.ReduceOp.SUM)  # sychronize with other processes
                # print("rank", utils.get_rank(), "finished all_reduce on ", metric_name, epoch_metrics[metric_name][split])


        # print("post reduce, on split", split, 'rank', utils.get_rank(), "has loss", epoch_metrics["loss"][split], utils.get_time())
        
        for metric_name,metric_result in epoch_metrics.items():  # log the average result to all_metrics
            all_metrics[metric_name][split].append((metric_result[split].mean() / utils.get_world_size()).item())

        # print("final loss was ", all_metrics["loss"][split][-1], utils.get_rank())
        # print("rank", utils.get_rank(), "finished on split", split)
        utils.barrier()
    net.train()
    oprint("finished evaluation at", utils.get_time())

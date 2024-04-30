import torch
import torch.distributed as dist
from tqdm import tqdm
import torch.nn.functional as F

from . import utils
from .utils import oprint, rprint

@torch.no_grad()
def accuracy(logits, targets):
    """Calculate the accuracy of the model on the given batch.
    Args:
        logits: Tensor of shape [B, seq_len, vocab_size]
        targets: Tensor of shape [B, seq_len]
    Returns:
        Tensor: scalar accuracy of the model on the given batch"""
    top_logits = torch.argmax(logits, dim=-1)  # [B, seq_len]
    return (top_logits == targets).float().mean()


@torch.no_grad()
def evaluate(net, dsets, exp_config, all_metrics):
    """Evaluate the model on the given dataset. 
    Args:
        net: The model to evaluate
        dsets: {split1: Dataset, split2: Dataset, ...}, where split1 is "train", "val", or "test"
        exp_config: Experiment configuration
        all_metrics: {metric_name: {split1: [val1, val2, ...], split2: [...], ...}, where val1, val2 
                are the values of the metric at each evaluation step
    Returns:
        None. all_metrics is modified in place to contain the results of the evaluation.
        """
    net.eval()
    # set up tensors for sychronization
    # TODO: maybe this should just be a [num_metrics, num_eval] tensor? 
    epoch_metrics = {k: {split: torch.zeros(exp_config.num_eval).to(net.device) 
                         for split in dsets} 
                     for k in all_metrics}
    
    oprint("starting evaluation at", utils.get_time())

    for split in dsets:
        for i, sample in tqdm(enumerate(dsets[split].dataloader())):
            if i >= exp_config.num_eval:  # so that we don't iterate over the entire training set while estimating tr_loss
                break
            if isinstance(sample, dict):  # handle commavq dataset
                sample = sample['xy'].transpose(0,1) # [B, 2, seq_len] -> [2, B, seq_len] so that we can split x and y

            x,y = [el.to(net.device, non_blocking=True) for el in sample]  # transfer to device

            logits, loss = net(x, y)  # match nanoGPT order of return

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
        
        if exp_config.ddp:  # synchronize the metrics across processes
            for metric_name in epoch_metrics.keys():  # do this since you shouldn't iterate over a thing you are modifying
                dist.all_reduce(epoch_metrics[metric_name][split], op=dist.ReduceOp.SUM)  # sychronize with other processes
        
        for metric_name,metric_result in epoch_metrics.items():  # log the average result to all_metrics
            all_metrics[metric_name][split].append((metric_result[split].mean() / utils.get_world_size()).item())

        utils.barrier()  # make sure all processes have finished sychronizing before moving on to the next split
    net.train()
    oprint("finished evaluation at", utils.get_time())

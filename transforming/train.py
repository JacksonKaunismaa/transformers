from tqdm import tqdm, trange
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os.path as osp
import pickle
import itertools
import wandb
import numpy as np
import dataclasses
import os
import time
# from torch.profiler import profile, record_function, ProfilerActivity

from .config_objects import ExperimentCfg
from .network import Transformer
from . import utils
from . import metrics
from . import net_utils
from .utils import rprint, oprint

def run_experiment(dsets, proj_name, resume_path, exp_config: ExperimentCfg, sample_method, 
                   log_wandb=True, resume=True):
    """Runs a single training experiment. This function is the main entry point for training a model. It handles
    all the details of setting up the model, optimizer, and scheduler, and running the training loop. It also
    handles logging to wandb and saving checkpoints.
    
    Args:
        dsets: dict of Dataset objects, containing the training and evaluation datasets
        proj_name: str, the name of the wandb project to log to
        resume_path: str, the path to a resume file, if resuming from a checkpoint
        exp_config: ExperimentCfg, the configuration object for the experiment
        sample_method: callable, a function that generates samples from the model for logging to wandb
        log_wandb: bool, whether to log to wandb
        resume: bool, whether to resume from a checkpoint
    """
    # no_init=True => don't initialize weights yet
    # since the resume path could contain a different config, so we need to load the config from the resume file first
    net = Transformer(exp_config, dsets["train"].cfg, no_init=True) 
    resumer = net_utils.Resumer(resume_path, exp_config.job_id)  # creates resumer object, for saving and loading checkpoints
    resumer.load(config=exp_config)  # load config from resume file (or don't if the resume file doesn't exist)
    net.initialize_architecture()  # create layers based on updated exp_config
    
    if log_wandb and utils.get_rank() == 0:  # create wandb session
        run_obj = wandb.init(project=proj_name,   # resume=True => we search for a a resume file and open it
                             id=resumer.wandb_run_id if resume else None,   # wandb_run_id is None if resume=False
                             resume="must" if resume and resumer.wandb_run_id else None,  # wandb_run_id is not None iff resume=True,
                             name=resumer.name,
                             config={**dataclasses.asdict(exp_config),          # and an existing resume file was found
                                     **dataclasses.asdict(dsets["train"].cfg)})
        # if wandb_run_id wasn't set before (either resume=False or no resume file found), set it now for sure
        resumer.wandb_run_id = run_obj.id  # type: ignore 

    oprint("model save path is ", resumer.model_save_path, "")
    device_id = utils.get_local_rank() # % utils.get_world_size()  # modulus to ensure contiguous?
    device = f"{utils.get_device_type()}:{device_id}"   # get the device string for the current process
    net.to(device)

    if exp_config.compile:  # should do compile after DDP due to extra optimizations that can be applied
        rprint("Compiling model...")  # in practice, its slower though
        net = torch.compile(net)

    if exp_config.ddp:
        rprint("worker", utils.get_rank(), "has", device_id, "World size is ", utils.get_world_size(), "dev_cnt",
            torch.cuda.device_count(), "dev_id", device_id, device, "local_rank", utils.get_local_rank(), ", connected to:"
            os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
        net = DDP(net, device_ids=[device_id], output_device=device_id)

    scaler, sched, optim = utils.get_raw(net).get_optim()  # type:ignore

    if resume:
        map_location = device  # make sure we load to the right device
        load_success = resumer.load(net=utils.get_raw(net), map_location=map_location, update_model_save_path=True, #type:ignore
                                    scaler=scaler, sched=sched, optim=optim)
        if not load_success:
            print("Attempted to load from checkpoint, but failed, aborting")
            return
        for gr in optim.param_groups:  # if the config loaded from the resume file changes the weight decay, update it
            if gr['weight_decay'] > 0:
                gr['weight_decay'] = exp_config.weight_decay

    rprint("weight decays", [gr['weight_decay'] for gr in optim.param_groups if 'weight_decay' in gr])
    oprint("Config is:", exp_config)

    # do the actual training
    train_result = train(net, resumer, scaler, sched, optim, exp_config, dsets, log_wandb, sample_method)
    
    if log_wandb and utils.get_rank() == 0:
        wandb.finish()

    if exp_config.ddp:
        dist.destroy_process_group()  # adds some overhead, but probably not a big deal? (when doing hyperparam optimization)

    return train_result


def train(net, resumer, scaler, scheduler, optimizer, exp_config: ExperimentCfg, dsets, log_wandb, sample_method):
    """Runs the training loop for a single experiment. This function is responsible for running the training loop,
    evaluating the model, logging to wandb, and saving checkpoints. It also handles the logic for early stopping based
    on the best loss.
    
    Args:
        net: Transformer, the model to train
        resumer: Resumer, the object for saving and loading checkpoints
        scaler: GradScaler, the gradient scaler
        scheduler: Scheduler, the learning rate scheduler
        optimizer: Optimizer, the optimizer
        exp_config: ExperimentCfg, the configuration object for the experiment
        dsets: dict of Dataset objects, containing the training and evaluation datasets (eg. {'train': Dataset, 'eval': Dataset})
        log_wandb: bool, whether to log to wandb
        sample_method: callable, a function that generates samples from the model for logging to wandb
    Returns:
        dict of {metric_name: {split: [metric_values]}}: the metrics for the training run
    """
    # TODO: consider changing this weird iterated dictionary thing into something clearer
    # maybe another class or only track the current epoch and do it with a {"loss", "perp", "accu"}
    # then you don't have to compute "shortened" names, the code is simpler. Another option is to 
    # put the "list" on the outside so that each entry in the list is a dict of {metric_name: value} mappings.
    # the advantage is you don't have to any extra work for extracting the most recent epoch's results.
    all_metrics = {k: dict(train=[], eval=[]) for k in ["loss", "perplexity", "accuracy"]}

    tr_loader = dsets["train"].dataloader()
    non_ddp_net = utils.get_raw(net)  # slight hack to get at the underlying Transformer object consistently

    # num tokens in an "epoch" 
    epoch_tokens = exp_config.grad_accum_steps * exp_config.batch_size * exp_config.block_size * exp_config.train_steps  

    rprint("Num tokens in an epoch:", epoch_tokens)
    rprint("Effective tokens per batch:", epoch_tokens//exp_config.train_steps)

    assert (exp_config.grad_accum_steps % utils.get_world_size() == 0)
    grad_accum_steps = exp_config.grad_accum_steps // utils.get_world_size() 
    rprint("running with accum_step", grad_accum_steps)

    while (curr_iter := scheduler.last_epoch) < exp_config.total_steps:
        start = utils.get_time()
        # rprint("after starting time", start, "mem alloc", torch.cuda.memory_summary())
        for s in trange(exp_config.train_steps): # tqdm + range

            for i in range(grad_accum_steps):  # compute gradients for a single macro-batch
                sample = next(tr_loader)  # TODO: maybe move getting the next sample to after the forward pass, to overlap better
                
                if isinstance(sample, dict):  # handle commavq dataset
                    sample = sample['xy'].transpose(0,1)  # switch batch and x vs. y dimension
                
                x,y = [el.to(net.device, non_blocking=True) for el in sample]
                
                if exp_config.ddp:   # only bother doing sync when doing the very last .backward() before an optimizer step
                    net.require_backward_grad_sync = ((i+1) % grad_accum_steps == 0)
                batch_loss = net(x, y)[1] / grad_accum_steps  # net returns (logits, loss_value) tuple
                
                scaler.scale(batch_loss).backward()  # accumulate scaled gradients
                
            scaler.unscale_(optimizer)  # unscale so that we can clip propely
            torch.nn.utils.clip_grad_norm_(net.parameters(), exp_config.grad_clip) # clip gradients
            scaler.step(optimizer)  # step otimizer and scaler with unscaled gradients
            scaler.update()  # update scale parameter
            optimizer.zero_grad(set_to_none=True)  # supposedly better on memory than .zero_grad()
            scheduler.step()  # update LR

        epoch_time = utils.get_time() - start
        metrics.evaluate(net, dsets, exp_config, all_metrics)
        curr_iter = scheduler.last_epoch
        short_name_metrics = {f"{split[:2]}_{metric_name[:4]}": result[-1]  # shorten names of metrics for pretty printing
                              for metric_name,results in all_metrics.items()  # read as an nested for-loop with the
                                for split,result in results.items()}              # shown indentation
        epoch_summary = f'Iter {curr_iter}: ' + " ".join(f"{name}: {val:.4f}" for name,val in short_name_metrics.items())
        if utils.get_rank() == 0:  # only 1 process needs to do this
            if log_wandb:
                rand_samples = sample_method(non_ddp_net, dsets, curr_iter)
                wandb.log({**short_name_metrics,
                           **rand_samples,  # named version of the "generate samples" concept, will just be {str: wandb_obj}, len=1
                           "lr": optimizer.param_groups[0]["lr"],
                           "tok_time": epoch_time/(epoch_tokens/utils.get_world_size()),
                           "step": curr_iter
                        })
            print(epoch_summary)
        rprint("checking whether to save, best was", non_ddp_net.best_loss, "last loss was", all_metrics["loss"]["eval"][-1])
        if all_metrics["loss"]["eval"][-1] < non_ddp_net.best_loss or np.isnan(all_metrics["loss"]["eval"][-1]):  
            non_ddp_net.best_loss = all_metrics["loss"]["eval"][-1]
            rprint("updating best loss to", non_ddp_net.best_loss)
            # update best loss so that all ranks have consistent .best_loss (and have a consistent view of when to save)
            # need to disable the rank check here since sharded optimizer requires all ranks to consolidate
            resumer.save(net=non_ddp_net, scaler=scaler, optim=optimizer, sched=scheduler)
                
    return all_metrics



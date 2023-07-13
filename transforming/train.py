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

from . import config_objects
from . import network
from . import utils
from . import metrics
from . import net_utils
from .utils import rprint

def run_experiment(dsets, proj_name, resume_path, exp_config: config_objects.ExperimentCfg, sample_method, 
                   log_wandb=True, resume=True):
    net = network.Transformer(exp_config, dsets["train"].cfg, no_init=True)
    resumer = net_utils.Resumer(resume_path, net, resume=resume)  # updates exp_config and net.cfg if resume file found
    net.initialize_architecture()  # create layers based on updated exp_config
    
    if log_wandb and utils.get_rank() == 0:
        run_obj = wandb.init(project=proj_name,   # resume=True => we search for a a resume file and open it
                             id=resumer.wandb_run_id if resume else None,   # wandb_run_id is None if resume=False
                             resume="must" if resume and resumer.wandb_run_id else None,  # wandb_run_id is not None iff resume=True
                             config={**dataclasses.asdict(exp_config),          # and an existing resume file was found
                                     **dataclasses.asdict(dsets["train"].cfg)})
        # if wandb_run_id wasn't set before (either resume=False or no resume file found), set it now for sure
        resumer.wandb_run_id = run_obj.id  # type: ignore 

    print("model save path is ", resumer.model_save_path)
    device_id = utils.get_local_rank() # % utils.get_world_size()  # modulus to ensure contiguous?
    device = f"{utils.get_device_type()}:{device_id}"
    # #rprint("setting device to", exp_config.device, net.cfg.device)
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        # torch.cuda.set_device(device_id)  # so that NCCL knows we are using different devices
    print(net)
    net = net.to(device)
    # #rprint("setting device to", exp_config.device, net.cfg.device)

    if exp_config.compile:
        print("Compiling model...")
        net = torch.compile(net)

    if exp_config.ddp:
        rprint("worker", utils.get_rank(), "has", device_id, "World size is ", utils.get_world_size(), "dev_cnt",
            torch.cuda.device_count(), "dev_id", device_id, device, "localr", utils.get_local_rank(),
            os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"], "bad")
        net = DDP(net, device_ids=[device_id], output_device=device_id)
        # [print(utils.get_local_rank(), torch.cuda.memory_allocated(n)) for n in range(torch.cuda.device_count())]

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model"):
    #         for i, sample in enumerate(dsets["train"].dataloader()):
    #             # print(utils.get_rank(), "has", sample[0][0])
    #             x,y = [el.to(net.device, non_blocking=True) for el in sample]
    #             net(x, y)
    #             if i > 50:
    #                 break
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=10))
    # return

    # #rprint("setting device to", exp_config.device, net.cfg.device)
    scaler, sched, optim = utils.get_raw(net).get_optim()

    if resume:
        map_location = device  # make sure we load to the right device
        load_success = resumer.load(map_location, update_model_save_path=True,  # after this .load(), exp_config won't be changed
                                    scaler=scaler, sched=sched, optim=optim)  # again, so its OK to do grad_accum_steps modifications
        if not load_success:
            print("Attempted to load from checkpoint, but failed, aborting")
            return
    # #rprint("setting device to", exp_config.device, net.cfg.device)

        
    if utils.get_rank() == 0:
        print("Config is:", exp_config)

    # loss_res = {"eval":[]}
    # utils.barrier()
    # start_time = utils.get_time()
    # evaluate(net, dsets, exp_config, loss_res)
    # print(loss_res, utils.get_time() - start_time, utils.get_rank())
    # dist.destroy_process_group()
    # return
    # print("waititng to log network params alone")
    # for _ in range(180):
    #     time.sleep(1)
    # print("done logging network params usage alon")


    train_result = train(net, resumer, scaler, sched, optim, exp_config, dsets, log_wandb, sample_method)
    
    if log_wandb and utils.get_rank() == 0:
        wandb.finish()

    if exp_config.ddp:
        dist.destroy_process_group()  # adds some overhead, but probably not a big deal? (when doing hyperparam optimization)

    return train_result


def train(net, resumer, scaler, scheduler, optimizer, exp_config: config_objects.ExperimentCfg, dsets, log_wandb, sample_method):
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
        #rprint("starting time", start)
        for _ in trange(exp_config.train_steps):

            for i in range(grad_accum_steps):  # compute gradients for a single macro-batch
                sample = next(tr_loader)  # TODO: maybe move getting the next sample to after the forward pass, to overlap better
                if isinstance(sample, dict):  # handle commavq dataset
                    sample = sample['xy'].transpose(0,1)  # switch batch and x vs. y dimension
                # rprint(sample.shape, sample.device)
                # rprint(len([el for el in sample]))
                x,y = [el.to(net.device, non_blocking=True) for el in sample]
                if exp_config.ddp:   # only bother doing sync when doing the very last .backward() before an optimizer step
                    net.require_backward_grad_sync = ((i+1) % grad_accum_steps == 0)
                batch_loss = net(x, y)[0] / grad_accum_steps  # net returns (loss_value, logits) tuple
                scaler.scale(batch_loss).backward()  # accumulate scaled gradients

            scaler.unscale_(optimizer)  # unscale so that we can clip propely
            torch.nn.utils.clip_grad_norm_(net.parameters(), exp_config.grad_clip)  # type: ignore
            scaler.step(optimizer)  # already gradients are unscaled (as they should be), this won't double unscale them
            scaler.update()  # update scale parameter
            optimizer.zero_grad(set_to_none=True)  # supposedly better on memory
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
        #rprint("checking whether to save, best was", non_ddp_net.best_loss, "last loss was", all_metrics["loss"]["eval"][-1])
        if all_metrics["loss"]["eval"][-1] < non_ddp_net.best_loss or all_metrics["loss"]["eval"][-1].isnan().any(): # type: ignore  
            non_ddp_net.best_loss = all_metrics["loss"]["eval"][-1]
            # if utils.get_rank() == 0:  # put rank check here so that all ranks have consistent .best_loss
            # need to disable the rank check here since sharded optimizer requires all ranks to consolidate
            resumer.save(scaler=scaler, optim=optimizer, sched=scheduler)
            # non_ddp_net.save_model_state_dict(scaler=scaler, optim=optimizer, sched=scheduler)
            # ie. save the nan model, but stop training
            if all_metrics["loss"]["eval"][-1].isnan().any():  # type: ignore
                print("detected nan failure, aborting training...")
                return
                # utils.barrier()  # wrapper around dist.barrier() that is a no-op if no ddp
                
            
    return all_metrics



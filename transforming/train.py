from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os.path as osp
import pickle
import itertools
import wandb
import numpy as np
import dataclasses
import time
import os

from . import config_objects
from . import network
from . import data
from . import utils

def run_experiment(dsets, proj_name, ckpt_path, exp_config: config_objects.ExperimentCfg, extend=False, log_wandb=False):
    if log_wandb and utils.get_rank() == 0:
        wandb.init(project=proj_name, config={**dataclasses.asdict(exp_config), 
                                              **dataclasses.asdict(dsets["train"].cfg)})
    
    net = network.Transformer(ckpt_path, exp_config, dsets["train"].cfg).to(dsets["train"].cfg.device)

    if exp_config.compile:
        print("Compiling model...")
        net = torch.compile(net)

    if exp_config.ddp:
        # [print(utils.get_local_rank(), torch.cuda.get_device_properties(n)) for n in range(torch.cuda.device_count())]
        assert (exp_config.grad_accum_steps % utils.get_world_size() == 0)
        exp_config.grad_accum_steps //= utils.get_world_size() # can write to config since wandb config is already set?       
        device_id = utils.get_local_rank() # % utils.get_world_size()  # modulus to ensure contiguous?
        # sort of hacky, so we only set the train dset_cfg, so we just need to ensure that we only use "train's" dset_cfg
        # from within the train function
        dsets['train'].cfg.device = f"{dsets['train'].cfg.device}:{device_id}"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        # torch.cuda.set_device(device_id)  # so that NCCL knows we are using different devices

        print("worker", utils.get_rank(), "has", device_id, "World size is ", utils.get_world_size(), "dev_cnt",
              torch.cuda.device_count(), "dev_id", device_id, dsets['train'].cfg.device, "localr", utils.get_local_rank(),
              os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"], "bad")
        net = DDP(net, device_ids=[device_id], output_device=device_id)
        [print(utils.get_local_rank(), torch.cuda.memory_allocated(n)) for n in range(torch.cuda.device_count())]
    for sample in dsets["train"].dataloader():
        print(utils.get_rank(), "has", sample[0][0])
        x,y = [el.to(dsets["train"].cfg.device, non_blocking=True) for el in sample]
        print(net(x, y).item())
        return


    scaler, sched, optim = utils.get_raw(net).get_optim()

    if extend:
        map_location = dsets['train'].cfg.device  # make sure we load to the right device
        utils.get_raw(net).load_model_state_dict(map_location, scaler=scaler, sched=sched, optim=optim)

    train_result = train(net, scaler, sched, optim, exp_config, dsets, log_wandb)
    
    if log_wandb and utils.get_rank() == 0:
        wandb.finish()

    if exp_config.ddp:
        dist.destroy_process_group()  # adds some overhead, but probably not a big deal? (when doing hyperparam optimization)

    return train_result


# def run_experiments(dsets, proj_name, ckpt_dir, hyperparams, prob_dists=None, search_type="grid", num_rand=20):
#     assert search_type in ["grid", "random"]
#     assert all(k in config_objects.ExperimentCfg().__dataclass_fields__ for k in hyperparams)
#     if prob_dists is None:
#         prob_dists = {}

#     log_path = osp.join(ckpt_dir, "log.pkl")
#     if osp.exists(log_path):
#         with open(log_path, "rb") as p:
#             saved = pickle.load(p)
#         #if saved["dset_cfg"] != dsets["train"].cfg: #or saved["hyperparams"] != hyperparams:
#             #print("Found existing log at path with different config than specified")
#             #if not override:  # override: set this to ignore the dset_config equality check
#             #    return
#         train_results, test_results = saved["train_results"], saved["test_results"]
#     else:
#         train_results, test_results = {}, {}

#     model_name = "_".join(f"{{{k}}}" for k in hyperparams) + ".ckpt"
#     model_path = osp.join(ckpt_dir, model_name)

#     log_dict = {"hyperparams": hyperparams,
#                 #"dset_cfg": train_dset.cfg,
#                 "train_results": train_results,
#                 "test_results": test_results}

#     hyp_keys, hyp_choices = list(hyperparams.keys()), list(hyperparams.values())
#     experiment_iter = itertools.product(*hyp_choices) if search_type == "grid" else range(num_rand)
#     for i, item in enumerate(experiment_iter):
#         if search_type == "grid":  # here, "item" is the actual selections for the hyperparameters
#             hyp_dict = dict(zip(hyp_keys, item))
#         elif search_type == "random":  # here "item" is just an integer
#             hyp_dict = {}
#             for k,choices in hyperparams.items():
#                 prob_dist = prob_dists.get(k)
#                 if isinstance(choices, dict):
#                     choices = list(choices.keys())                
#                 hyp_dict[k] = np.random.choice(choices, p=prob_dist)
        
#         # use the named choices version of hyp_dict
#         name = model_path.format(**hyp_dict)  # guarantees that the format specified in name matches the actual hyperparams

#         # fetch the values associated with the named choices
#         for k,choice in hyp_dict.items():
#             if isinstance(choice, str):  # assume that if a hyperparameter takes a string value, it's a named choice
#                 hyp_dict[k] = hyperparams[k][choice]

#         # use the value-only version of hyp_dict
#         exp_config = config_objects.ExperimentConfig(**hyp_dict)
#         if exp_config in train_results:
#             print("Already completed experiment for", name)
#             continue
#         print("Running experiment for", name, "experiment", i+1)

#         train_result, test_result = run_experiment(dsets, proj_name, name, exp_config)
        
#         train_results[exp_config] = train_result
#         test_results[exp_config] = test_result

#         with open(log_path, "wb") as p:
#             pickle.dump(log_dict, p)


@torch.no_grad()  # won't each rank get a slightly different number here?
def evaluate(net, dsets, exp_config, all_losses):
    net.eval()
    losses = {split:[] for split in dsets}
    for split, dset in dsets.items():
        for i, sample in tqdm(enumerate(dset.dataloader())):
                x,y = [el.to(dsets["train"].cfg.device, non_blocking=True) for el in sample]
                losses[split].append(net(x, y).item())
                if i > exp_config.num_eval:  # so that we don't iterate over the entire training set while estimating tr_loss
                    break
        all_losses[split].append(sum(losses[split])/len(losses[split]))
    net.train()


def train(net, scaler, scheduler, optimizer, exp_config: config_objects.ExperimentCfg, dsets, log_wandb):
    losses = dict(train=[], eval=[])
    tr_loader = data.make_infinite(dsets["train"].dataloader())
    non_ddp_net = utils.get_raw(net)  # slight hack to get at the underlying Transformer object consistently

    epoch_tokens = exp_config.grad_accum_steps * dist.get_world_size() * \
                   exp_config.batch_size * exp_config.block_size * exp_config.num_train  # num tokens in an "epoch"

    while (curr_iter := scheduler.state_dict()["last_epoch"]) < exp_config.total_iters:
        start = utils.get_time()
        for i in tqdm(range(exp_config.num_train)):
            sample = next(tr_loader)  # TODO: maybe move getting the next sample to after the forward pass, to overlap better
            x,y = [el.to(dsets["train"].cfg.device, non_blocking=True) for el in sample]
            if exp_config.ddp:
                net.require_backward_grad_sync = ((i+1) % exp_config.grad_accum_steps == 0)
            batch_loss = net(x, y) / exp_config.grad_accum_steps
            scaler.scale(batch_loss).backward()  # accumulate scaled gradients
            if (i+1) % exp_config.grad_accum_steps == 0:  # only actually set gradients after grad_accum steps
                scaler.unscale_(optimizer)  # unscale so that we can clip propely
                torch.nn.utils.clip_grad_norm_(net.parameters(), exp_config.grad_clip)
                scaler.step(optimizer)  # already gradients are unscaled (as they should be), this won't double unscale them
                scaler.update()  # update scale parameter
                optimizer.zero_grad(set_to_none=True)  # supposedly better on memory
                scheduler.step()  # update LR

        epoch_time = utils.get_time() - start

        evaluate(net, dsets, exp_config, losses)
        curr_iter = scheduler.state_dict()["last_epoch"]
        epoch_summary = f'Iter {curr_iter}: ' + " ".join(f"{split[:2]}_loss: {loss[-1]:.4f}" for split,loss in losses.items())
        if utils.get_rank() == 0:  # only 1 process needs to do this
            if log_wandb:
                wandb.log({"tr_loss": losses["train"][-1], 
                        "va_loss": losses["eval"][-1],
                        "lr": optimizer.param_groups[0]["lr"],
                        "step": curr_iter,
                        "tok_time": epoch_time/epoch_tokens})
            print(epoch_summary)

        if losses["eval"][-1] < non_ddp_net.best_loss:  # maybe dont save every epoch if loss improves?
            non_ddp_net.best_loss = losses["eval"][-1]
            if utils.get_rank() == 0:  # put rank check here so that all ranks have consistent .best_loss
                non_ddp_net.save_model_state_dict(scaler=scaler, optim=optimizer, sched=scheduler)
                # utils.barrier()  # wrapper around dist.barrier() that is a no-op if no ddp
                
            
    return losses



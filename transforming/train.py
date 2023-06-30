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
from torch.profiler import profile, record_function, ProfilerActivity

from . import config_objects
from . import network
from . import data
from . import utils
from . import metrics
from .utils import rprint

def run_experiment(dsets, proj_name, ckpt_path, exp_config: config_objects.ExperimentCfg, 
                   extend=0, log_wandb=False, resume_id=""):
    if log_wandb and utils.get_rank() == 0:
        wandb.init(project=proj_name, id=resume_id if resume_id else None, resume="must" if resume_id else None,
                   config={**dataclasses.asdict(exp_config),
                           **dataclasses.asdict(dsets["train"].cfg)})

    #if exp_config.ddp:
        # [print(utils.get_local_rank(), torch.cuda.get_device_properties(n)) for n in range(torch.cuda.device_count())]
    print("model save path is ", ckpt_path)
    net = network.Transformer(ckpt_path, exp_config, dsets["train"].cfg) # type: ignore

    assert (exp_config.grad_accum_steps % utils.get_world_size() == 0)
    exp_config.grad_accum_steps //= utils.get_world_size() # can write to config since wandb config is already set?       
    device_id = utils.get_local_rank() # % utils.get_world_size()  # modulus to ensure contiguous?
    # sort of hacky, so we only set the train dset_cfg, so we just need to ensure that we only use "train's" dset_cfg
    # from within the train function
    #exp_config.device = f"{exp_config.device}:{device_id}"
    rprint("running with accum_step", exp_config.grad_accum_steps)
    device = f"{utils.get_device_type()}:{device_id}"
    # #rprint("setting device to", exp_config.device, net.cfg.device)
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        # torch.cuda.set_device(device_id)  # so that NCCL knows we are using different devices

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
    # #rprint("setting device to", exp_config.device, net.cfg.device)

    if extend:  # set extend to the slurm job id of the run that generated it so we can acces the checkpoint
        map_location = device  # make sure we load to the right device
        actual_ckpt_dir = osp.dirname(osp.realpath(ckpt_path))
        old_ckpt_dir = osp.join(osp.dirname(actual_ckpt_dir), str(extend))  # go into the different job id's checkpoint dir
        old_ckpt_path = osp.join(old_ckpt_dir, osp.basename(ckpt_path))
        load_success = utils.get_raw(net).load_model_state_dict(map_location, path=old_ckpt_path, 
                                                                scaler=scaler, sched=sched, optim=optim)
        if not load_success:
            print("Attempted to load from checkpoint, but failed, aborting")
            return

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



def train(net, scaler, scheduler, optimizer, exp_config: config_objects.ExperimentCfg, dsets, log_wandb):
    # TODO: consider changing this weird iterated dictionary thing into something clearer
    # maybe another class or only track the current epoch and do it with a {"loss", "perp", "accu"}
    # then you don't have to compute "shortened" names, the code is simpler. Another option is to 
    # put the "list" on the outside so that each entry in the list is a dict of {metric_name: value} mappings.
    # the advantage is you don't have to any extra work for extracting the most recent epoch's results.
    all_metrics = {k: dict(train=[], eval=[]) for k in ["loss", "perplexity", "accuracy"]}

    tr_loader = data.make_infinite(dsets["train"].dataloader())
    non_ddp_net = utils.get_raw(net)  # slight hack to get at the underlying Transformer object consistently

    epoch_tokens = exp_config.grad_accum_steps * utils.get_world_size() * \
                    exp_config.batch_size * exp_config.block_size * exp_config.train_steps  # num tokens in an "epoch"

    # all_sentences = []  # do this entirely in python since wandb really hates being able to update artifacts
    # sentence_artifact_name = "rand_sentences{}"  # do this since it would probably be too much io sadly
    # sentence_batch = 
    # #rprint("exp_cfg device", exp_config.device, "net device", non_ddp_net.cfg.device)
    while (curr_iter := scheduler.last_epoch) < exp_config.total_steps:
        start = utils.get_time()
        #rprint("starting time", start)
        for _ in trange(exp_config.train_steps):

            for i in range(exp_config.grad_accum_steps):  # compute gradients for a single macro-batch
                sample = next(tr_loader)  # TODO: maybe move getting the next sample to after the forward pass, to overlap better
                x,y = [el.to(net.device, non_blocking=True) for el in sample]
                if exp_config.ddp:   # only bother doing sync when doing the very last .backward() before an optimizer step
                    net.require_backward_grad_sync = ((i+1) % exp_config.grad_accum_steps == 0)
                batch_loss = net(x, y)[0] / exp_config.grad_accum_steps  # net returns (loss_value, logits) tuple
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
                # all_sentences.extend([[curr_iter, sent] for sent in utils.sample_random_sentences(net, dsets, exp_config)])
                # print(wandb.run.logged_artifacts())
                rand_sentences = [[curr_iter, sent] for sent in utils.sample_random_sentences(non_ddp_net, dsets, exp_config)]
                wandb.log({**short_name_metrics,
                        "lr": optimizer.param_groups[0]["lr"],
                        "tok_time": epoch_time/epoch_tokens,
                        "rand_sentences": wandb.Table(columns=["step", "sentence"], data=rand_sentences),
                        "step": curr_iter
                        })#, step=curr_iter)  # TODO: fix this for future projectss
            print(epoch_summary)
        #rprint("checking whether to save, best was", non_ddp_net.best_loss, "last loss was", all_metrics["loss"]["eval"][-1])
        if all_metrics["loss"]["eval"][-1] < non_ddp_net.best_loss or all_metrics["loss"]["eval"][-1].isnan().any(): # type: ignore  
            non_ddp_net.best_loss = all_metrics["loss"]["eval"][-1]
            # if utils.get_rank() == 0:  # put rank check here so that all ranks have consistent .best_loss
            # need to disable the rank check here since sharded optimizer requires all ranks to consolidate
            non_ddp_net.save_model_state_dict(scaler=scaler, optim=optimizer, sched=scheduler)
            # ie. save the nan model, but stop training
            if all_metrics["loss"]["eval"][-1].isnan().any():  # type: ignore
                print("detected nan failure, aborting training...")
                return
                # utils.barrier()  # wrapper around dist.barrier() that is a no-op if no ddp
                
            
    return all_metrics



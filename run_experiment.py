import torch
import os.path as osp
import torch.distributed as dist
import argparse
import torch.multiprocessing as mp
import os
import torch._dynamo
import socket
import datasets as hf
hf.disable_caching()
# import torch._functorch

# import transformers
from transforming.train import run_experiment
from transforming.config_objects import ExperimentCfg, DatasetCfg, CommaVQDatasetCfg
from transforming.text_task.data import IdxDataset
from transforming.commavq.data import CommaVQDataset
from transforming import utils
from transforming.utils import rprint
from transforming import net_utils

torch.backends.cuda.matmul.allow_tf32 = True # type: ignore
torch.backends.cudnn.allow_tf32 = True # type: ignore
#torch._functorch.config.functionalize_rng_ops=True # type: ignore
# torch._dynamo.config.verbose = True # type: ignore
# torch._dynamo.config.log_level = logging.INFO # type: ignore

def main(local_rank, args):
    if local_rank is not None:
        global_rank = args.rank * args.local_world_size + local_rank
        global_world_size = args.local_world_size * args.nnodes
        print("global stuff", global_world_size, global_rank)
        print("port and addr", os.environ["MASTER_PORT"], os.environ["MASTER_ADDR"], utils.get_rank())
        dist.init_process_group(backend="nccl", rank=global_rank, world_size=global_world_size)
        torch.cuda.set_device(local_rank)  # so that nccl knows we are only using that specific device
        os.environ["LOCAL_RANK"] = str(local_rank)  # so that we can local rank access later (arguably bad design)
    
    rprint("hi from proc, world size is", utils.get_world_size(), torch.cuda.device_count())
    for v in ["NCCL_ALGO", "NCCL_PROTO", "NCCL_BUFFSIZE", "NCCL_SOCKET_NTHREADS", "NCCL_NSOCKS_PERTHREAD"]:
        print(v, os.environ.get(v, "not found"))
    exp_config = ExperimentCfg(vec_size=1408,
                            n_layer=22,
                            n_heads=11,
                            lr_max=2e-4,
                            lr_min=1e-7,
                            t_decay=50_000,  # there are 15B tokens roughly, so this means we iterate over the data about 1x
                            total_steps=50_000,  # since our model uses ~500M parameters, this is the right-ish amount
                            block_size=2624,  # 41*64
                            batch_size=1,
                            grad_accum_steps=144,  # 2*48
                            train_steps=550, # num macro batches, need to make this small for pre-emption purposes
                            num_eval=300,  # num micro batches
                            dtype="float16",
                            compile=True,
                            zero=True,
                            checkpointing=False,
                            normalizer_type="RMSNorm",
                            rmsnorm_p=0.07,
                            layer_norm_posn="pre",
                            posn_embed_type="rotary",
                            rotary_dim=64,
                            flash=True,
                            mqa_attn=True,
                            learnable_unembed=True,
                            job_id=args.id,
                            max_generation_len=2624,
                            num_sample=0
                            )
    if args.dry:  # if dry run, overwrite config with dry_run config
        exp_config = exp_config.get_dry()

    exp_config.ddp = local_rank is not None

    
    utils.barrier()
    if args.task == "text":
        data_dir = "/scratch/ssd004/scratch/jackk/1-billion-word-language-modeling-benchmark-r13output"
        task_name = "transformer-experiments-google-1-billion"
        dset_config = DatasetCfg(dataset_path=data_dir,
                                 chunk_size=20)

        datasets = dict(train=IdxDataset("train.bin", exp_config, dset_config),
                        eval=IdxDataset("eval.bin", exp_config, dset_config))
        sample_method = net_utils.wandb_sample_random_sentences

    elif args.task == "commavq":
        task_name = "transformer-experiments-commavq"
        dset_config = CommaVQDatasetCfg(decoder_path="/scratch/ssd004/scratch/jackk/commavq/commavq/models/decoder.onnx")
        load_ds = hf.load_dataset("commaai/commavq", 
                                data_dir="/scratch/ssd004/scratch/jackk/commavq/", 
                                cache_dir="/scratch/ssd004/scratch/jackk/commavq")
        
        datasets = dict(train=CommaVQDataset([load_ds[str(i)] for i in range(40)], exp_config, dset_config.replace(split_ranks=True)), # type:ignore
                        eval=CommaVQDataset(load_ds['40'], exp_config, dset_config, load_decoder=False)) # type:ignore
        sample_method = net_utils.wandb_sample_random_videos



    run_experiment(datasets, task_name, args.resume_path if not args.dry else "resumes/dry.ckpt",   # type:ignore
                   exp_config, sample_method, log_wandb=True, resume=True)                  # type:ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int, help="global base rank")
    parser.add_argument('--local-world-size', default=1, type=int, help="all gpu processes")
    parser.add_argument('--nnodes', default=1, type=int, help="number of compute-nodes (local_world_size * nnodes = world_size)")
    parser.add_argument('--dry', action="store_true", help="set this flag to run with a very small network, for debugging purposes")
    parser.add_argument('--id', default=0, type=int, help="Slurm job id (for true location of checkpoint)")
    parser.add_argument('--resume-path', help="Path of resume file where model cfg, save path, and wandb_run_id will be stored")
    parser.add_argument('--task', help="Select which type of dataset to use", choices=["commavq", "text"], default='text')
    args = parser.parse_args()
    print(args)
    print("cuda state", torch.cuda.is_available(), torch.cuda.device_count())
    print("running on node ", socket.gethostname())
    if args.nnodes > 1 or args.local_world_size > 1:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = utils.get_random_unused_port()
        print("about to spawn with addr", os.environ["MASTER_ADDR"], "port", os.environ["MASTER_PORT"])
        mp.spawn(main, args=(args,), nprocs=args.local_world_size) # type: ignore
    else:
        main(None, args)

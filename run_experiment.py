import torch
import os.path as osp
import pickle
import torch.distributed as dist
import argparse
import torch.multiprocessing as mp
import os

# import transformers
from transforming.train import run_experiment
from transforming.config_objects import ExperimentCfg, DatasetCfg
from transforming.data import IdxDataset
from transforming import utils

# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(local_rank, args, data_dir):
    if local_rank is not None:
        global_rank = args.rank * args.local_world_size + local_rank
        global_world_size = args.local_world_size * args.nnodes
        dist.init_process_group(backend="nccl", rank=global_rank, world_size=global_world_size)
        torch.cuda.set_device(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)

    print("hi from proc", utils.get_rank(), "world size is", utils.get_world_size(), torch.cuda.device_count())
    exp_config = ExperimentCfg(vec_size=768,
                            n_layer=12,
                            n_heads=12,
                            lr_max=6e-4,
                            lr_min=1e-7,
                            block_size=1024,
                            batch_size=4,
                            grad_accum_steps=64,
                            num_train=4_000,
                            num_eval=300,
                            dtype="float16",
                            compile=False,
                            ddp=local_rank is not None,
                            )

    dset_config = DatasetCfg(dataset_path=data_dir,
                            device=device,
                            num_workers=args.num_workers
                            )

    datasets = dict(train=IdxDataset("train.bin", exp_config, dset_config),
                    eval=IdxDataset("eval.bin", exp_config, dset_config))
    
    # if exp_config.ddp:
    #     print("inited")
    #     dist.init_process_group("nccl")
    print("post hi from proc", utils.get_rank(), "world size is", utils.get_world_size(), torch.cuda.device_count())

    run_experiment(datasets, "transformer-experiments-google-1-billion", "checkpoint/flash-medium-1-gpu-amp.ckpt", exp_config,  
               log_wandb=False, extend=False)


if __name__ == "__main__":
    data_dir = "/scratch/ssd004/scratch/jackk/1-billion-word-language-modeling-benchmark-r13output"

    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--rank', default=0, type=int, help="global base rank")
    parser.add_argument('--local-world-size', default=1, type=int, help="all gpu processes")
    parser.add_argument('--nnodes', default=1, type=int, help="number of compute-nodes (local_world_size * nnodes = world_size)")
    args = parser.parse_args()
    if args.nnodes > 0 or args.local_world_size > 0:
        mp.spawn(main, args=(args, data_dir), nprocs=args.local_world_size)
    else:
        main(None, args, data_dir)

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

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(local_rank, args, data_dir):
    if local_rank is not None:
        global_rank = args.rank * args.local_world_size + local_rank
        global_world_size = args.local_world_size * args.nnodes
        print("global stuff", global_world_size, global_rank)
        print("port and addr", os.environ["MASTER_PORT"], os.environ["MASTER_ADDR"], utils.get_rank())
        dist.init_process_group(backend="nccl", rank=global_rank, world_size=global_world_size)
        torch.cuda.set_device(local_rank)  # so that nccl knows we are only using that specific device
        os.environ["LOCAL_RANK"] = str(local_rank)  # so that we can local rank access later (arguably bad design)

    print("hi from proc", utils.get_rank(), "world size is", utils.get_world_size(), torch.cuda.device_count())
    for v in ["NCCL_ALGO", "NCCL_PROTO", "NCCL_BUFFSIZE", "NCCL_SOCKET_NTHREADS", "NCCL_NSOCKS_PERTHREAD"]:
        print(v, os.environ.get(v, "not found"))
    exp_config = ExperimentCfg(vec_size=1536,
                            n_layer=12,
                            n_heads=12,
                            lr_max=2e-4,
                            lr_min=1e-7,
                            block_size=1024,
                            batch_size=4,
                            grad_accum_steps=64,
                            num_train=4_000,
                            num_eval=300,
                            dtype="float16",
                            compile=True,
                            zero=True,
                            normalizer_type="RMSNorm",
                            rmsnorm_p=0.2,
                            )
    if args.dry:  # if dry run, overwrite config with dry_run config
        exp_config = exp_config.get_dry()

    exp_config.ddp = local_rank is not None

    dset_config = DatasetCfg(dataset_path=data_dir,
                            num_workers=args.num_workers
                            )

    datasets = dict(train=IdxDataset("train.bin", exp_config, dset_config),
                    eval=IdxDataset("eval.bin", exp_config, dset_config))
    

    utils.barrier()
    run_experiment(datasets, "transformer-experiments-google-1-billion", "checkpoint/large-multi-gpu-zero-rmsnorm.ckpt", exp_config,  
               log_wandb=True, extend=0)#9783411, resume_id="mqa0qyio")
#9783411
#9816078

if __name__ == "__main__":
    data_dir = "/scratch/ssd004/scratch/jackk/1-billion-word-language-modeling-benchmark-r13output"

    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--num-workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--rank', default=0, type=int, help="global base rank")
    parser.add_argument('--local-world-size', default=1, type=int, help="all gpu processes")
    parser.add_argument('--nnodes', default=1, type=int, help="number of compute-nodes (local_world_size * nnodes = world_size)")
    parser.add_argument('--dry', action="store_true", help="set this flag to run with a very small network, for debugging purposes")
    args = parser.parse_args()
    print(args)
    if args.nnodes > 1 or args.local_world_size > 1:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = utils.get_random_unused_port()
        mp.spawn(main, args=(args, data_dir), nprocs=args.local_world_size)
    else:
        main(None, args, data_dir)

import torch
import torch.utils.data
from torch.utils.data import Dataset, Sampler, DataLoader
import os.path as osp
import numpy as np
import copy


from ..config_objects import ExperimentCfg, DatasetCfg
from . import encoder
from .. import utils


def make_infinite(dataloader):
    while True:
        yield from dataloader


class IdxDataset(Dataset):  # this feels like it shouldn't work... (has to learn to ignore all context before EOS)
    """Concatenate all sentences in the dataset into one long sequence, and then sample from it. Sentences are separated by EOS tokens.
    This implementation will use np.memmap to map the index dataset, which comes from the output of build_dataset, into memory."""
    def __init__(self, fname, exp_cfg: ExperimentCfg, dset_cfg: DatasetCfg):
        self.data = np.memmap(osp.join(dset_cfg.dataset_path, fname), dtype=np.uint16, mode="r")#, shape=(data_size,))
        self.encoder = encoder.get_encoder(dset_cfg.dataset_path)

        self.exp_cfg = exp_cfg

        self.cfg = copy.copy(dset_cfg)
        self.cfg.vocab_size = int(np.ceil(len(self.encoder.idx_to_tok)/64)*64)  # pad by rounding up to nearest 64 for efficiency
        self.cfg.total_size = self.data.shape[0]
        
    def __len__(self):
        # return 5  # toggle this for a testing/debugging run
        return (self.cfg.total_size - self.exp_cfg.block_size - 1) // self.cfg.chunk_size
    
    def dataloader(self, num_workers=1):
        if self.exp_cfg.ddp:
            sampler = torch.utils.data.DistributedSampler(self, shuffle=True)
        else:
            sampler = torch.utils.data.RandomSampler(self, replacement=False)
        print("created sampler", utils.get_rank())
        return make_infinite(DataLoader(self, batch_size=self.exp_cfg.batch_size, 
                                        sampler=sampler, 
                                        pin_memory=True,
                                        num_workers=num_workers))#self.cfg.num_workers//2)  #//3 since the cpus aren't enough???

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_idx = idx * self.cfg.chunk_size  
        # due to the way len is defined, this shouldnt hit any IndexErrors
        x = torch.from_numpy(self.data[data_idx : data_idx+self.exp_cfg.block_size].astype(np.int32))  # minor space save
        y = torch.from_numpy(self.data[data_idx+1 : data_idx+self.exp_cfg.block_size+1].astype(np.int64))
        return x, y
    
def build_dataset(data_dir):  # helper function that should probably be moved into a script somewhere
    """Map a dataset of text files into a single long sequence of indices.
    This function will create two .bin files in the data_dir, one for training and one for evaluation.
    Inside of data_dir, it searches for two subdirectories, "training-monolingual.tokenized.shuffled" and 
    "heldout-monolingual.tokenized.shuffled", which contain the text files to be encoded. 
    Args:
        data_dir (str): A directory containing 2 subdirectories, "training-monolingual.tokenized.shuffled" and 
                        "heldout-monolingual.tokenized.shuffled", each of which contain text files.
    Returns:
        None, but writes two .bin files to data_dir.
    """
    import pickle
    from .encoder import get_encoder
    # create idx dataset .bin files if not already created
    if not osp.exists(osp.join(data_dir, "sizes.pkl")):
        enc = get_encoder(data_dir)

        data_sizes = {"train.bin": enc.encode_file_list(data_dir, "training-monolingual.tokenized.shuffled", "train.bin", 12),
                      "eval.bin": enc.encode_file_list(data_dir, "heldout-monolingual.tokenized.shuffled", "eval.bin", 12)}
    
        with open(osp.join(data_dir, "sizes.pkl"), "wb") as p:
            pickle.dump(data_sizes, p)
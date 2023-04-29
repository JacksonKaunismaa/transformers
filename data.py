import torch
from torch.utils.data import Dataset
import requests
from os import path as osp
import json
import numpy as np
import regex as re  # required since regular re does not support character classes \p
import multiprocessing as mp
import itertools
from tqdm import tqdm


def create_byte_mapper():
    # this seems not scrictly necessary, but basically it will map all possible bytes in [0,255] to a character
    # to represent it visually. This basically is just to make things look nicer (instead of having \x0b\x1d all
    # over the place, you will have a single (potentially strange) character to represent it)

    # these bytes all render fine interpreted as ASCII, so keep em
    byte_list = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    byte_reprs = byte_list[:]  # since these bytes look nice already we can map them to themselves

    n = 0
    for b in range(2**8):
        if b in byte_list:
            continue
        byte_list.append(b)
        byte_reprs.append(2**8+n)
        n += 1

    byte_reprs = [chr(c) for c in byte_reprs]
    return dict(zip(byte_list, byte_reprs))

def download_remote(url, fname):
    if not osp.exists(fname):
        response = requests.get(url)
        with open(fname, "wb") as f:
           f.write(response.content)


def batch_data(data, n): # itertools recipe
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(data)
    while batch := tuple(itertools.islice(it, n)):
        yield batch

class Encoder():  # or should it be called tokenizer
    def __init__(self, tok_to_idx, bpe_merges):
        # the full pipeline is byte_encoder -> bpe_encoder -> encoder, which in our code is
        # byte_encoder -> bpe_merger -> token_to_idx
        self.byte_encoder = create_byte_mapper()
        self.byte_decoder = {v:k for k,v in self.byte_encoder.items()}

        # give order on merges so that we know the order in which to merge on our words
        self.bpe_merge_order = {merge:i for i,merge in enumerate(bpe_merges)}

        self.tok_to_idx = tok_to_idx
        self.pad_token = len(tok_to_idx)
        self.eos_token = len(tok_to_idx)+1
        self.idx_to_tok = {v:k for k,v in tok_to_idx.items()}
        self.idx_to_tok[self.pad_token] = "<PAD>"  # byte decoder maps these to themselves, so its fine
        self.idx_to_tok[self.eos_token] = "<EOS>"

        # the magic regex that splits a sentence into tokens in the coarsest way possible. The "matches" will
        # be punctation, full words, or word ending abbreviations.
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {} # cache

    def get_bigrams(self, token):  # token in the sense of "something having come from self.pat" + merging steps
        return set(zip(token, token[1:]))

    def bpe_merge(self, token):
        if token in self.cache:  # check cache first
            return self.cache[token]

        curr_token = list(token)  # we only do the string->list thing so the cache works

        while len(curr_token) > 1:  # if it's merged to a single token, we are done anyway
            bigrams = self.get_bigrams(curr_token)
            # print("\t", bigrams, token, "bigr", "tok")
            next_merge = min(bigrams, key=lambda bg: self.bpe_merge_order.get(bg, float("inf")))
            # print("\tdecided to merge", next_merge)
            if next_merge in self.bpe_merge_order:  # ie. there is a valid merge still left to do
                merged_word = []
                bigram_token_iter = zip(curr_token, curr_token[1:])
                for prev_tok, next_tok in bigram_token_iter:
                    # print("\t\tchecking for merge at", prev_tok, next_tok)
                    if (prev_tok, next_tok) == next_merge:
                        # print("\t\tmerge is good")
                        merged_word.append("".join(next_merge))
                        try:
                            next(bigram_token_iter) # skip by 2 spaces instead of 1 so that we don't double-append next_tok
                        except StopIteration:  # if we are already on the last one
                            break
                    else:
                        # print("\t\tno merge, moving along")
                        merged_word.append(prev_tok)
                else:   # if the last bigram wasn't merged, we would miss appending the very last token
                    merged_word.append(curr_token[-1])        # so make sure we don't miss it
                # print("\tword now is", merged_word)
                curr_token = merged_word
            else:
                break
        self.cache[token] = curr_token
        return curr_token

    def encode(self, text):
        try:
            tokens = re.findall(self.pat, text)
        except TypeError:
            print(text[:min(300, len(text))])
            raise
        idx_list = []
        for tok in tokens:
            unicode_tok = tok.encode("utf-8")
            pretty_tok = "".join(self.byte_encoder[b] for b in unicode_tok)  # string format for the cache
            # print(pretty_tok, "pretty")
            bpe_tok = self.bpe_merge(pretty_tok)
            # print(bpe_tok, "bpe merged")

            idx_tok = [self.tok_to_idx[t] for t in bpe_tok]
            idx_list += idx_tok
        return idx_list

    def process_one(self, text):
        enc_tokens = self.encode(text)
        enc_tokens.append(self.eos_token)
        return np.asarray(enc_tokens, dtype=np.uint16)

    def encode_file_list(self, out_fname, in_fnames):
        all_lines = []
        for fname in in_fnames:
            with open(fname, "r") as f_in:
                all_lines.extend(f_in.readlines())
        print("Read all files in, beginning to encode...")
        # note that there is no benefit to doing this asynchronously since we need the full data size
        # to be able to start writing? (is this even true?)
        with mp.Pool(7) as pool:  # use the heuristic/default chunksize used in Pool.map()
            enc_lines = pool.map(self.process_one, all_lines, chunksize=len(all_lines)//(7*4))  # returns list of numpy arrays
        #enc_lines = [self.encode(line) for line in tqdm(all_lines)]
        self.write_file(out_fname, enc_lines)


    def write_file(self, out_fname, enc_lines):
        print("Calculating size...")
        size = sum([e.shape[0] for e in tqdm(enc_lines)])
        print("Preparing to write", size, "values to", out_fname)
        out_f = np.memmap(out_fname, dtype=enc_lines[0].dtype, mode="w+", shape=(size,))
        idx = 0
        for batch in tqdm(batch_data(enc_lines, 1024)): # write in groups of 1024
            linearized_data = np.concatenate(batch)
            out_f[idx: idx+len(linearized_data)] = linearized_data
            idx += len(linearized_data)
        out_f.flush()

    def decode(self, idx_list):
        # if tok is not in idx_to_tok its probably a control token like PAD or STOP or something else, so ignore it
        tokens = [self.idx_to_tok[idx] for idx in idx_list if idx in self.idx_to_tok]
        text = []
        for tok in tokens:
            text_tok = [chr(self.byte_decoder[c]) for c in tok]
            text += text_tok
        # print(text)
        return "".join(text)


class TextDataset(Dataset):
    def __init__(self, fname):
        data = np.memmap(fname, dtype=np.uint16, mode="r")
        self.encoder = get_encoder()
        vocab_size = len(self.encoder.tok_to_idx)
        self.pad_token = vocab_size
        self.vocab_size = vocab_size + 1  # including pad token

    def collate_fn(self, samples):
        # 0 maps to ! for now 
        # shouldnt really be a problem? our sequences end with \n so we can treat that as STOP
        padded = torch.nn.utils.rnn.pad_sequence(samples, 
                                                 padding_value=self.pad_token, 
                                                 batch_first=True)[:, :self.block_size] 
        return padded[:, :-1], padded[:, 1:]


    def __getitem__(self, idx):

        text = self.strings[idx]
        indices = self.encoder.encode(text)
        indices = torch.tensor(indices, dtype=torch.uint16)

        # inputs = indices[:-1]
        # targets = indices[1:]

        return indices  # do that in collate_fn instead

    def __len__(self):
        return len(self.strings)


def get_encoder():
    # use openai token_to_idx and bpe_merge
    download_remote("https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json", "encoder.json")
    download_remote("https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe", "vocab.bpe")

    with open("encoder.json", "r") as f:
        token_to_idx = json.load(f)
    with open("vocab.bpe", "r") as f:
        bpe_merge_list = [tuple(x.strip().split()) for x in f.readlines()]

    return Encoder(token_to_idx, bpe_merge_list[1:-1])

import mmap
import re
import numpy as np
import pickle
from argparse import ArgumentParser
from datetime import datetime
import os

parser = ArgumentParser()


def preprocess(all_games: list, move_dir:str):
    def process_row(row):
        if len(row) == 0 or row[0] != "1":
            return []
        return [re.sub(r"[\+#]", "", word) for word in row.split(" ") if len(re.findall(r"[^0-9abcdefghxBKNPQRO\-\+#]", word)) ==0][:-1]

    for filename in all_games:
        with open(filename, 'r') as fh:
            i = 0
            for line in fh:
                with open(f"{move_dir}/moves_train.txt", "a") as f:
                        with open(f"{move_dir}/moves_validate.txt", "a") as g:
                            game = process_row(line)
                            if len(game) > 0:
                                if i % 20 == 0:
                                    g.write(" ".join(game) + "\n")
                                else:
                                    f.write(" ".join(game) + "\n")

                                i += 1
                                if i % 100000 == 0 and i > 0:
                                    print("[%s]: Number of games processed: %s" % (datetime.now(), i))

def tokenize(move_dir):
    unique_moves = set("\n")
    total_tokens_train = 0
    total_tokens_validate = 0

    with open(f"{move_dir}/moves_train.txt", "r") as f:
        for line in f:
            moves = [re.sub("\n", "", move) for move in line.split(" ")]
            [unique_moves.add(move) for move in moves]
            total_tokens_train += len(moves) + 1

    with open(f"{move_dir}/moves_validate.txt", "r") as f:
        for line in f:
            moves = [re.sub("\n", "", move) for move in line.split(" ")]
            [unique_moves.add(move) for move in moves]
            total_tokens_validate += len(moves) + 1

    unique_moves = list(unique_moves)
    vocab_size = len(unique_moves)
    stoi = { ch:i for i,ch in enumerate(unique_moves) }
    itos = { i:ch for i,ch in enumerate(unique_moves) }

    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    row_index = 0
    list_index = 0

    train_file = np.memmap(f'{move_dir}/train.bin', dtype=np.uint16, mode='w+', shape=(total_tokens_train,))
    validate_file = np.memmap(f'{move_dir}/val.bin', dtype=np.uint16, mode='w+', shape=(total_tokens_validate,))
    del train_file
    del validate_file

    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(f'{move_dir}/meta.pkl', 'wb') as f:
        pickle.dump(meta, f)


    with open(f"{move_dir}/moves_train.txt", "r") as f:
        for line in f:
            train_file = np.memmap(f'{move_dir}/train.bin', dtype=np.uint16, mode='r+', shape=(total_tokens_train,))
            moves = [re.sub("\n", "", move) for move in line.split(" ")]
            for token in encode(moves + ["\n"]):
                train_file[list_index]=token
                list_index += 1
            row_index += 1
            del train_file

    row_index = 0
    list_index = 0
            
    with open(f"{move_dir}/moves_validate.txt", "r") as f:
        for line in f:
            validate_file = np.memmap(f'{move_dir}/val.bin', dtype=np.uint16, mode='r+', shape=(total_tokens_validate,))
            moves = [re.sub("\n", "", move) for move in line.split(" ")]
            for token in encode(moves + ["\n"]):
                validate_file[list_index]=token
                list_index += 1
            row_index += 1
            del validate_file


parser.add_argument("function")
parser.add_argument("--files", nargs='+')
parser.add_argument("--outDir")
args = parser.parse_args()

if args.function == "tokenize":
    tokenize(args.outDir)
elif args.function == "preprocess":
    os.makedirs(args.outDir, exist_ok=True)
    preprocess(args.files, args.outDir)
else:
    raise BaseException("Invalid function")

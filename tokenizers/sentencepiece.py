"""
Source: https://github.com/PaccMann/paccmann_proteomics/blob/master/scripts/

"""

import argparse
import glob
import os
from os.path import join
from loguru import logger
import re
import numpy as np
import pandas as pd

from tokenizers import SentencePieceBPETokenizer

parser = argparse.ArgumentParser()

parser.add_argument(
    '--path_data_dir',
    default=None,
    metavar='path',
    type=str,
    required=True,
    help='Path to the input folder containing the data set',
)
parser.add_argument(
    '--out',
    #default='./',
    type=str,
    required=True,
    help='Path to the output directory, where the files will be saved',
)
parser.add_argument(
    '--sequence_file',
    default=None,
    type=str,
    required=True,
    help='Path of the input file that containts the AA sequences; if it does not exist setting create_sequence_file to True will create one',
)
parser.add_argument(
    '--chunk_size',
    default=10**6,
    type=int,
    help='The number of sequences to load each time',
)
parser.add_argument(
    '--name', 
    default='sentencepiece.json', 
    type=str, 
    help='The name of the output vocab file'
)
parser.add_argument(
    '--vocab_size', 
    default=30000, 
    type=int,
    required=False,
    help='Vocabulary size',
)
parser.add_argument(
    '--test_encoding', 
    default=False, 
    type=bool,
    required=False,
    help="Quickly test the trained tokenizer",
)
parser.add_argument(
    '--limit_alphabet',
    default=1000,
    type=int,
    help='The size of alphabet character set (e.g., for English, |alphabet|=26)',
)

args = parser.parse_args()

#create output folder
os.makedirs(args.out, exist_ok=True)

#hardcoding it to the preprocessing script output for now
file = args.path_data_dir + "/uniprot_sprot_100.csv"
if not file:
    logger.info(f'Input folder: {args.path_data_dir} is empty')
    exit(1)

#Get sequence data in chunnks since we have millions of sequences
if not (os.path.isfile(join(args.out, args.sequence_file))):
    seq_data = []
    for c in pd.read_csv(file, encoding='utf8', sep='\n', skiprows=0, chunksize = args.chunk_size):
        for line in c.values:
            aa_seq = line[0].split(';')[-1].strip(',')
            aa_seq = re.sub('\s+','', aa_seq)
            seq_data.append(aa_seq)

#    with open(file, encoding='utf8') as f:
#        lines = f.readlines()
#        for line in lines[1:]:
#            aa_seq = line.split(';')[-1].strip(',')
#            aa_seq = re.sub('\s+','', aa_seq)
#            seq_data.append(aa_seq)

    np.savetxt(join(args.out, args.sequence_file), np.array(seq_data), encoding='utf8', fmt='%s')

# Initialize an empty tokenizer
tokenizer = SentencePieceBPETokenizer(add_prefix_space=True)

#Train tokenizer 
# add a better error here to indicate user has create_sequence_file False without a sequence file available
tokenizer.train(
    join(args.out, args.sequence_file),
    vocab_size=args.vocab_size,
    min_frequency=2,
    show_progress=True,
    special_tokens=['<unk>'],
    limit_alphabet=args.limit_alphabet
)

# Save the file
tokenizer.save(join(args.out, args.name))

# Test encoding
if args.test_encoding:
    logger.info('Tokens and their ids from ByteLevelBPETokenizer with GFP protein sequence: \n MSKGEE LFTGVVPILVELDGDVNGHKFSVSGEGEG DAT')
    encoded = tokenizer.encode('MSKGEE LFTGVVPILVELDGDVNGHKFSVSGEGEG DAT')
    logger.info(encoded.tokens)
    logger.info(encoded.ids)
    logger.info('done!')
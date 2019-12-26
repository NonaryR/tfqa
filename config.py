import os
import sys
sys.path.append("src")

from functools import partial
from pathlib import Path
import argparse
import ujson as json

import torch
from apex import amp

from src.utils import mklogs
from src.data_utils import convert_data, collate_fn, eval_collate_fn
from src.train import run

from src.nets import BertForQuestionAnswering
from src.spanbert.tokenization import BertTokenizer
from src.optimizers import get_optimizer, get_scheduler
from src.losses import loss_fn

from typing import List, Dict, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# acc steps 0, max seq len (512, 800), max_question_len 32, learning rate

def main(args):
    
    logs_dir = mklogs(args)
    
    DATA_DIR = Path('../data/')
    TRAIN_DATA = DATA_DIR / args.train_file
    DEV_DATA = DATA_DIR / args.val_file

    do_lower_case = 'uncased' in args.bert_model
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=do_lower_case)
    
    model = BertForQuestionAnswering.from_pretrained(args.bert_model)
    model = model.cuda()
    
    optimizer = get_optimizer(model.named_parameters(), args.learning_rate)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
    model = torch.nn.DataParallel(model)

    convert_fn = partial(convert_data, tokenizer=tokenizer, max_seq_len=args.max_seq_len,
                         max_question_len=args.max_question_len, doc_stride=args.doc_stride)    

    num_train_optimization_steps = int(args.n_epochs * args.train_size / args.batch_size / args.accumulation_steps)
    num_warmup_steps = int(num_train_optimization_steps * args.warmup)   
    scheduler = get_scheduler(optimizer, num_warmup_steps, num_train_optimization_steps)

    run(model,
        TRAIN_DATA,
        DEV_DATA,
        logs_dir,
        optimizer,
        scheduler,
        loss_fn,
        collate_fn,
        eval_collate_fn,
        convert_fn,
        args.n_epochs,
        args.batch_size,
        args.accumulation_steps,
        args.chunksize,
        args.train_size,
        args.val_size,
        eval_=args.do_eval)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train bertQA on NQ')

    parser.add_argument('-t',
                        '--train_file',
                        type=str,
                        help='train dataset name',
                        default="train.jsonl")

    parser.add_argument('-val',
                        '--val-file',
                        type=str,
                        help='val dataset name',
                        default="dev.jsonl")

    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        help='seed',
                        default=609)

    parser.add_argument('-chz',
                        '--chunksize',
                        type=int,
                        help='chunking for json reader',
                        default=1000)

    parser.add_argument('-n_epochs',
                        '--n_epochs',
                        type=int,
                        help='number of epochs',
                        default=2)

    parser.add_argument('-l',
                        '--logs_dir',
                        type=str,
                        help="dir for logs, model, results",
                        default="logs")                    

    parser.add_argument('-ts',
                        '--train_size',
                        type=int,
                        help='wc -l on train.jsonl',
                        default=292006)

    parser.add_argument('-vs',
                        '--val_size',
                        type=int,
                        help='wc -l on dev.jsonl',
                        default=15367)

    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        help='batch size for nn.DataParallel',
                        default=28)

    parser.add_argument('-acs',
                        '--accumulation_steps',
                        type=int,
                        help='acc steps for bigger batch aka ...',
                        default=4)

    parser.add_argument('-msl',
                        '--max_seq_len',
                        type=int,
                        help='[answer + question][:max seq len]',
                        default=384)

    parser.add_argument('-mql',
                        '--max_question_len',
                        type=int,
                        help='[question][:max seq len]',
                        default=64)

    parser.add_argument('-ds',
                        '--doc_stride',
                        type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.",
                        default=128)

    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        help="rate for learning =)",
                        default=2e-5)

    parser.add_argument('-wrm',
                        '--warmup',
                        type=float,
                        help="",
                        default=0.05)

    parser.add_argument('-b',
                        '--bert_model',
                        type=str,
                        help="which model",
                        default='bert-base-uncased')
    
    parser.add_argument('-de',
                        '--do_eval',
                        type=str,
                        default="do") 

    args = parser.parse_args()
    
    main(args)

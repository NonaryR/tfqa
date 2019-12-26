from dataclasses import dataclass
from typing import Callable, Dict, List, Generator, Tuple, Any
import pandas as pd
import re
import ujson as json
import numpy as np
import torch

@dataclass
class TestExample(object):
    example_id: int
    candidates: List[Dict]
    doc_start: int
    question_len: int
    tokenized_to_original_index: List[int]
    input_ids: List[int]

def convert_test_data(
    line: str,
    tokenizer, #: BertTokenizer,
    max_seq_len: int,
    max_question_len: int,
    doc_stride: int
) -> List[TestExample]:
    """Convert dictionary data into list of training data.

    Parameters
    ----------
    line : str
        Training data.
    tokenizer : transformers.BertTokenizer
        Tokenizer for encoding texts into ids.
    max_seq_len : int
        Maximum input sequence length.
    max_question_len : int
        Maximum input question length.
    doc_stride : int
        When splitting up a long document into chunks, how much stride to take between chunks.
    """

    # model input
    data = json.loads(line)
    doc_words = data['document_text'].split()
    question_tokens = tokenizer.tokenize(data['question_text'])[:max_question_len]

    # tokenized index of i-th original token corresponds to original_to_tokenized_index[i]
    # if a token in original text is removed, its tokenized index indicates next token
    original_to_tokenized_index = []
    tokenized_to_original_index = []
    all_doc_tokens = []  # tokenized document text
    
    for i, word in enumerate(doc_words):
        original_to_tokenized_index.append(len(all_doc_tokens))
        if re.match(r'<.+>', word):  # remove paragraph tag
            continue
        sub_tokens = tokenizer.tokenize(word)
        for sub_token in sub_tokens:
            tokenized_to_original_index.append(i)
            all_doc_tokens.append(sub_token)

    # make sure at least one object in `examples`
    examples = []
    max_doc_len = max_seq_len - len(question_tokens) - 3  # [CLS], [SEP], [SEP]

    # take chunks with a stride of `doc_stride`
    for doc_start in range(0, len(all_doc_tokens), doc_stride):
        doc_end = doc_start + max_doc_len
        
        doc_tokens = all_doc_tokens[doc_start:doc_end]
        input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + doc_tokens + ['[SEP]']
        examples.append(
            TestExample(
                example_id=data['example_id'],
                candidates=data['long_answer_candidates'],
                # annotations=annotations,
                doc_start=doc_start,
                question_len=len(question_tokens),
                tokenized_to_original_index=tokenized_to_original_index,
                input_ids=tokenizer.convert_tokens_to_ids(input_tokens),
                # start_position=start,
                # end_position=end,
                # class_label=label,
                # line=data,
        ))

    return examples
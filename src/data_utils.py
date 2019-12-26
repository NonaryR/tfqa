from dataclasses import dataclass
from typing import Callable, Dict, List, Generator, Tuple, Any
import pandas as pd
import re
import ujson as json
import numpy as np
import torch

@dataclass
class Example(object):
    example_id: int
    candidates: List[Dict]
    annotations: Dict
    doc_start: int
    question_len: int
    tokenized_to_original_index: List[int]
    input_ids: List[int]
    start_position: int
    end_position: int
    class_label: str
    # line: Any


def convert_data(
    line: str,
    tokenizer, #: BertTokenizer,
    max_seq_len: int,
    max_question_len: int,
    doc_stride: int
) -> List[Example]:
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

    def _find_short_range(short_answers: List[Dict]) -> Tuple[int, int]:
        answers = pd.DataFrame(short_answers)
        start_min = answers['start_token'].min()
        end_max = answers['end_token'].max()
        return start_min, end_max

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

    # model output: (class_label, start_position, end_position)
    annotations = data['annotations'][0]
    
    if annotations['yes_no_answer'] in ['YES', 'NO']:
        class_label = annotations['yes_no_answer'].lower()
        start_position = annotations['long_answer']['start_token']
        end_position = annotations['long_answer']['end_token']
    
    elif annotations['short_answers']:
        class_label = 'short'
        start_position, end_position = _find_short_range(annotations['short_answers'])
    
    elif annotations['long_answer']['candidate_index'] != -1:
        class_label = 'long'
        start_position = annotations['long_answer']['start_token']
        end_position = annotations['long_answer']['end_token']
    
    else:
        class_label = 'unknown'
        start_position = -1
        end_position = -1

    # convert into tokenized index
    if start_position != -1 and end_position != -1:
        start_position = original_to_tokenized_index[start_position]
        end_position = original_to_tokenized_index[end_position]

    # make sure at least one object in `examples`
    examples = []
    max_doc_len = max_seq_len - len(question_tokens) - 3  # [CLS], [SEP], [SEP]

    # take chunks with a stride of `doc_stride`
    for doc_start in range(0, len(all_doc_tokens), doc_stride):
        doc_end = doc_start + max_doc_len
        # if truncated document does not contain annotated range
        if not (doc_start <= start_position and end_position <= doc_end):
            start, end, label = -1, -1, 'unknown'
        else:
            start = start_position - doc_start + len(question_tokens) + 2
            end = end_position - doc_start + len(question_tokens) + 2
            label = class_label

        assert -1 <= start < max_seq_len, f'start position is out of range: {start}'
        assert -1 <= end < max_seq_len, f'end position is out of range: {end}'

        doc_tokens = all_doc_tokens[doc_start:doc_end]
        input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + doc_tokens + ['[SEP]']
        examples.append(
            Example(
                example_id=data['example_id'],
                candidates=data['long_answer_candidates'],
                annotations=annotations,
                doc_start=doc_start,
                question_len=len(question_tokens),
                tokenized_to_original_index=tokenized_to_original_index,
                input_ids=tokenizer.convert_tokens_to_ids(input_tokens),
                start_position=start,
                end_position=end,
                class_label=label,
                # line=data,
        ))

    return examples



def eval_collate_fn(examples: List[Example]) -> Tuple[List[torch.Tensor], List[Example]]:
    # input tokens
    max_len = max([len(example.input_ids) for example in examples])
    tokens = np.zeros((len(examples), max_len), dtype=np.int64)
    token_type_ids = np.ones((len(examples), max_len), dtype=np.int64)
    for i, example in enumerate(examples):
        row = example.input_ids
        tokens[i, :len(row)] = row
        token_type_id = [0 if i <= row.index(102) else 1
                         for i in range(len(row))]  # 102 corresponds to [SEP]
        token_type_ids[i, :len(row)] = token_type_id
    attention_mask = tokens > 0
    inputs = [torch.from_numpy(tokens),
              torch.from_numpy(attention_mask),
              torch.from_numpy(token_type_ids)]

    return inputs, examples


def collate_fn(examples: List[Example]) -> List[List[torch.Tensor]]:
    # input tokens
    max_len = max([len(example.input_ids) for example in examples])
    tokens = np.zeros((len(examples), max_len), dtype=np.int64)
    token_type_ids = np.ones((len(examples), max_len), dtype=np.int64)
    for i, example in enumerate(examples):
        row = example.input_ids
        tokens[i, :len(row)] = row
        token_type_id = [0 if i <= row.index(102) else 1
                         for i in range(len(row))]  # 102 corresponds to [SEP]
        token_type_ids[i, :len(row)] = token_type_id
    attention_mask = tokens > 0
    inputs = [torch.from_numpy(tokens),
              torch.from_numpy(attention_mask),
              torch.from_numpy(token_type_ids)]

    # output labels
    all_labels = ['long', 'no', 'short', 'unknown', 'yes']
    start_positions = np.array([example.start_position for example in examples])
    end_positions = np.array([example.end_position for example in examples])
    class_labels = [all_labels.index(example.class_label) for example in examples]
    start_positions = np.where(start_positions >= max_len, -1, start_positions)
    end_positions = np.where(end_positions >= max_len, -1, end_positions)
    labels = [torch.LongTensor(start_positions),
              torch.LongTensor(end_positions),
              torch.LongTensor(class_labels)]

    return [inputs, labels]

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas
from sklearn.metrics import f1_score

from typing import List, Dict
from tqdm import tqdm
from result import Result

def eval_model(
    model: nn.Module,
    valid_loaders: List[DataLoader],
    stage: str
) -> Dict[str, float]:
    """Compute validation score.
    
    Parameters
    ----------
    model : nn.Module
        Model for prediction.
    valid_loader : DataLoader
        Data loader of validation data.
    device : torch.device, optional
        Device for computation.
    
    Returns
    -------
    dict
        Scores of validation data.
        `long_score`: score of long answers
        `short_score`: score of short answers
        `overall_score`: score of the competition metric
    """
    model.cuda()
    model.eval()
    with torch.no_grad():
        result = Result()
        for valid_loader in tqdm(valid_loaders, desc=f"evaluate on {stage}"):
            for inputs, examples in valid_loader:
                input_ids, attention_mask, token_type_ids = inputs
                y_preds = model(input_ids.cuda(non_blocking=True),
                                attention_mask.cuda(non_blocking=True),
                                token_type_ids.cuda(non_blocking=True))
                
                start_preds, end_preds, class_preds = (p.detach().cpu() for p in y_preds)
                start_logits, start_index = torch.max(start_preds, dim=1)
                end_logits, end_index = torch.max(end_preds, dim=1)

                # span logits minus the cls logits seems to be close to the best
                cls_logits = start_preds[:, 0] + end_preds[:, 0]  # '[CLS]' logits
                logits = start_logits + end_logits - cls_logits  # (batch_size,)
                indices = torch.stack((start_index, end_index)).transpose(0, 1)  # (batch_size, 2)
                result.update(examples, logits.numpy(), indices.numpy(), class_preds.numpy())

    return result.score(), model.train(True)

# METRIC IMPLEMANTATION by KAGGLE
def in_shorts(row):
    return row['short_answer'] in row['short_answers']

def get_f1(answer_df,label_df):
    short_label =  (label_df['short_answers'] != '').astype(int)
    long_label =  (label_df['long_answer'] != '').astype(int)

    long_predict = np.zeros(answer_df.shape[0])
    long_predict[(answer_df['long_answer'] == label_df['long_answer']) & (answer_df['long_answer'] != '')] = 1
    long_predict[(label_df['long_answer'] == '') & (answer_df['long_answer'] != '')] = 1  # false positive

    short_predict = np.zeros(answer_df.shape[0])
    short_predict[(label_df['short_answers'] == '') & (answer_df['short_answer'] != '')] = 1  # false positive
    a = pd.concat([answer_df[['short_answer']],label_df[['short_answers']]], axis = 1)
    a['short_answers'] = a['short_answers'].apply(lambda x: x.split())
    short_predict[a.apply(lambda x: in_shorts(x), axis = 1) & (a['short_answer'] != '')] = 1

    long_f1 = f1_score(long_label.values,long_predict)
    short_f1 = f1_score(short_label.values,short_predict)
    micro_f1 = f1_score(np.concatenate([long_label,short_label]), np.concatenate([long_predict,short_predict]))
    return micro_f1, long_f1, short_f1


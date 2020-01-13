from collections import defaultdict
from data_utils import Example
import torch
import numpy as np
import pandas as pd
from typing import List, Generator, Dict

class Result(object):
    """Stores results of all test data.
    """
    
    def __init__(self):
        self.examples = {}
        self.results = {}
        self.best_scores = defaultdict(float)
        self.class_labels = ['LONG', 'NO', 'SHORT', 'UNKNOWN', 'YES']
        
    @staticmethod
    def is_valid_index(example: Example, index: List[int]) -> bool:
        """Return whether valid index or not.
        """
        start_index, end_index = index
        if start_index > end_index:
            return False
        if start_index <= example.question_len + 2:
            return False
        return True
        
    def update(
        self,
        examples: List[Example],
        logits: torch.Tensor,
        indices: torch.Tensor,
        class_preds: torch.Tensor
    ):
        """Update batch objects.
        
        Parameters
        ----------
        examples : list of Example
        logits : np.ndarray with shape (batch_size,)
            Scores of each examples..
        indices : np.ndarray with shape (batch_size, 2)
            `start_index` and `end_index` pairs of each examples.
        class_preds : np.ndarray with shape (batch_size, num_classes)
            Class predicition scores of each examples.
        """
        for i, example in enumerate(examples):
            if self.is_valid_index(example, indices[i]) and \
               self.best_scores[example.example_id] < logits[i]:
                self.best_scores[example.example_id] = logits[i]
                self.examples[example.example_id] = example
                self.results[example.example_id] = [
                    example.doc_start, indices[i], class_preds[i]]

    def _generate_predictions(self) -> Generator[Dict, None, None]:
        """Generate predictions of each examples.
        """
        for example_id in self.results.keys():
            doc_start, index, class_pred = self.results[example_id]
            
            example = self.examples[example_id]
            tokenized_to_original_index = example.tokenized_to_original_index
            if doc_start + index[1] >= len(tokenized_to_original_index):
                continue
            
            short_start_index = tokenized_to_original_index[doc_start + index[0]]
            short_end_index = tokenized_to_original_index[doc_start + index[1]]

            long_start_index = -1
            long_end_index = -1
            for candidate in example.candidates:
                if candidate['start_token'] <= short_start_index and \
                   short_end_index <= candidate['end_token']:
                    long_start_index = candidate['start_token']
                    long_end_index = candidate['end_token']
                    break
            yield {
                'example': example,
                'long_answer': [long_start_index, long_end_index],
                'short_answer': [short_start_index, short_end_index],
                'yes_no_answer': class_pred
            }

    def end(self) -> Dict[str, Dict]:
        """Get predictions in submission format.
        """
        preds =[] # {}
        for pred in self._generate_predictions():
            example = pred['example']
            long_start_index, long_end_index = pred['long_answer']
            short_start_index, short_end_index = pred['short_answer']
            class_pred = pred['yes_no_answer']

            long_answer = f'{long_start_index}:{long_end_index}' if long_start_index != -1 else np.nan
            short_answer = f'{short_start_index}:{short_end_index}'
            class_pred = self.class_labels[class_pred.argmax()]
            short_answer += ' ' + class_pred if class_pred in ['YES', 'NO'] else ''
            
            preds.append({"example_id": f"{example.example_id}_long", "PredictionString": long_answer}) 
            preds.append({"example_id": f"{example.example_id}_short", "PredictionString": short_answer}) 
        
        return pd.DataFrame(preds)

    def score(self) -> Dict[str, float]:
        """Calculate score of all examples.
        """

        def _safe_divide(x: int, y: int) -> float:
            """Compute x / y, but return 0 if y is zero.
            """
            if y == 0:
                return 0.
            else:
                return x / y

        def _compute_f1(answer_stats: List[List[bool]]) -> float:
            """Computes F1, precision, recall for a list of answer scores.
            """
            has_answer, has_pred, is_correct = list(zip(*answer_stats))
            precision = _safe_divide(sum(is_correct), sum(has_pred))
            recall = _safe_divide(sum(is_correct), sum(has_answer))
            f1 = _safe_divide(2 * precision * recall, precision + recall)
            return f1

        long_scores = []
        short_scores = []
        for pred in self._generate_predictions():
            example = pred['example']
            long_pred = pred['long_answer']
            short_pred = pred['short_answer']
            class_pred = pred['yes_no_answer']
            yes_no_label = self.class_labels[class_pred.argmax()]

            # long score
            long_label = example.annotations['long_answer']
            has_answer = long_label['candidate_index'] != -1
            has_pred = long_pred[0] != -1 and long_pred[1] != -1
            is_correct = False
            if long_label['start_token'] == long_pred[0] and \
               long_label['end_token'] == long_pred[1]:
                is_correct = True
            long_scores.append([has_answer, has_pred, is_correct])

            # short score
            short_labels = example.annotations['short_answers']
            class_pred = example.annotations['yes_no_answer']
            has_answer = yes_no_label != 'NONE' or len(short_labels) != 0
            has_pred = class_pred != 'NONE' or (short_pred[0] != -1 and short_pred[1] != -1)
            is_correct = False
            if class_pred in ['YES', 'NO']:
                is_correct = yes_no_label == class_pred
            else:
                for short_label in short_labels:
                    if short_label['start_token'] == short_pred[0] and \
                       short_label['end_token'] == short_pred[1]:
                        is_correct = True
                        break
            short_scores.append([has_answer, has_pred, is_correct])

        long_score = _compute_f1(long_scores)
        short_score = _compute_f1(short_scores)
        return {
            'long_score': np.round(long_score, 5),
            'short_score': np.round(short_score, 5),
            'overall_score': np.round((long_score + short_score) / 2, 5)
        }

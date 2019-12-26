import os
import itertools
from multiprocessing import Pool
import ujson as json
import random

from pandas.io.json.json import JsonReader
from typing import Callable, List

import numpy as np
from torch.utils.data import Dataset
from data_utils import Example
from inference import TestExample

CPU_COUNT = os.cpu_count()

class JsonChunkReader(JsonReader):
    """JsonReader provides an interface for reading in a JSON file.
    """
    
    def __init__(
        self,
        filepath_or_buffer: str,
        convert_data: Callable[[str], List[Example]],
        chunksize: int = 2000,
        orient: str = None,
        typ: str = 'frame',
        dtype: bool = None,
        convert_axes: bool = None,
        convert_dates: bool = True,
        keep_default_dates: bool = True,
        numpy: bool = False,
        precise_float: bool = False,
        date_unit: str = None,
        encoding: str = None,
        lines: bool = True,
        compression: str = None,
    ):
        # JsonChunkReader, self
        super().__init__(
            str(filepath_or_buffer),
            orient=orient, typ=typ, dtype=dtype,
            convert_axes=convert_axes,
            convert_dates=convert_dates,
            keep_default_dates=keep_default_dates,
            numpy=numpy, precise_float=precise_float,
            date_unit=date_unit, encoding=encoding,
            lines=lines, chunksize=chunksize,
            compression=compression
        )
        self.convert_data = convert_data
        
    def __next__(self):
        lines = list(itertools.islice(self.data, self.chunksize))
        if lines:
            with Pool(CPU_COUNT) as p:
                obj = p.map(self.convert_data, lines)
            return obj

        self.close()
        raise StopIteration


class TextDataset(Dataset):

    def __init__(self, examples: List[Example], aug=False):
        self.examples = examples
        self.aug = aug
        
    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index):
        
        annotated = list(
            filter(lambda example: example.class_label != 'unknown', self.examples[index]))
        
        item = annotated if len(annotated) > 0 else self.examples[index]
        
        if self.aug == True:
            return random.choice(item)
        else:
            return item[0]


class TextDatasetOriginal(Dataset):
    r"""Dataset for [TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering).
    
    Parameters
    ----------
    examples : list of Example
        The whole Dataset.
    """
    
    def __init__(self, examples: List[Example]):
        self.examples = examples
        
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, index):
        annotated = list(
            filter(lambda example: example.class_label != 'unknown', self.examples[index]))
        if len(annotated) == 0:
            return random.choice(self.examples[index])
        return random.choice(annotated)


class TestTextDataset(Dataset):
    r"""Dataset for [TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering).
    
    Parameters
    ----------
    examples : list of Example
        The whole Dataset.
    """
    
    def __init__(self, examples: List[TestExample]):
        self.examples = examples
        
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, index):
        
        for item in self.examples[index]:
            return item
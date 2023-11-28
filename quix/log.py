import torch
import os
import json

from numbers import Number
from typing import Dict, Any, Sequence, Union, Mapping, Optional, Dict

StrDict = Dict[str, Any]

def _tostring(value: Any) -> Union[Number, str, Mapping]:
    if value is None:
        return 'None'
    if isinstance(value, dict):
        return {k: _tostring(v) for k, v in value.items()}
    if isinstance(value, Number) or isinstance(value, str):
        return value
    return str(value)

class AbstractLogger:

    def __init__(self, keywords:Sequence[str], *args, **kwargs):
        self.keywords = keywords

    def _filterfn(self, pair) -> bool:
        key, _ = pair
        if key in self.keywords:
            return True
        return False

    def filter_logs(self, **logging_kwargs) -> StrDict:
        return dict(filter(self._filterfn, logging_kwargs.items()))
    
    def __call__(self, **logging_kwargs) -> StrDict:
        return self.log(**self.filter_logs(**logging_kwargs))

    def log(self, **logging_kwargs) -> StrDict:
        return logging_kwargs
    

class LossLogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(['loss'])


class AccuracyLogger(AbstractLogger):

    def __init__(self, top_k:int, *args, **kwargs):
        super().__init__(['outputs', 'targets'])
        self.logname = f'Acc{top_k}'
        self.top_k = top_k

    def log(self, **logging_kwargs) -> StrDict:
        outputs = logging_kwargs['outputs']
        targets = logging_kwargs['targets']
        if not torch.is_tensor(outputs) and torch.is_tensor(targets):
            raise ValueError(
                'AccuracyLogger expects single output and target tensor. '
                f'Got {type(outputs)=} {type(targets)=}'
            )
        if not outputs.shape[0] == targets.shape[0]:
            raise ValueError(
                'AccuracyLogger expects input and output tensor of same batch dimension. '
                f'Got {outputs.shape=} {targets.shape=}'
            )
        nb = len(targets)
        if targets.ndim == 1:
            labelidx = targets[:,None]
        else:
            labelidx = targets.topk(1, dim=-1).indices

        acc = (outputs.topk(self.top_k, dim=-1).indices == labelidx).count_nonzero().div(nb)
        return {self.logname: acc}        


LoggerSequence = Sequence[AbstractLogger]


class BaseLogHandler:

    def __init__(
        self, 
        runid:str,
        project:str,
        loggers:LoggerSequence,
        root:Optional[str]=None,
    ):
        self.runid = runid
        self.project = project
        self.loggers = loggers
        if root is None:
            self.root = os.path.expanduser('~')
        else:
            self.root = root
        self.project_path = os.path.join(self.root, self.project)
        os.makedirs(self.project_path, exist_ok=True)
        self.file_path = os.path.join(self.project_path, f"{runid}.jsonl")

    def __call__(self, **logging_kwargs):
        timestamp = logging_kwargs['time']
        log_entry = {
            'time': timestamp, 
            **{key: value for logger in self.loggers for key, value in logger(**logging_kwargs).items()}
        }
        mode = 'a' if os.path.isfile(self.file_path) else 'w'
        with open(self.file_path, mode) as log_file:
            log_file.write(json.dumps(log_entry))
            log_file.write('\n')

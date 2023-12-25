import torch
import os
import json
import time

from typing import Dict, Any, Sequence, Dict

StrDict = Dict[str, Any]

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
    

class ProgressLogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(['epoch', 'iteration', 'training'])


class LossLogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(['loss'])


class LRLogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(['last_lr'])


class AccuracyLogger(AbstractLogger):

    def __init__(self, top_k:int, *args, **kwargs):
        super().__init__(['outputs', 'targets'])
        self.logname = f'Acc{top_k}'
        self.top_k = top_k

    def unpack(self, name, dict):
        val = dict.get(name, [])
        if isinstance(val, tuple) or isinstance(val, list):
            if len(val) != 1:
                raise ValueError(
                    f'AccuracyLogger expects single output and target tensor. '
                    f'Got {type(val)} of length {len(val)}.'
                )
            return self.unpack(name, {name:val[0]})
        return val

    def log(self, **logging_kwargs) -> StrDict:
        outputs = self.unpack('outputs', logging_kwargs)
        targets = self.unpack('targets', logging_kwargs)       
        if not torch.is_tensor(outputs) or not torch.is_tensor(targets):
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
    

class DeltaTimeLogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(['time'])
        self.logname = 'timedelta'
        self.prev_time = time.time()

    def log(self, **logging_kwargs):
        cur_time = logging_kwargs.get('time', None)
        if cur_time is None:
            return {}
        delta_time = cur_time - self.prev_time
        self.prev_time = cur_time
        return {self.logname: delta_time}
    

LoggerSequence = Sequence[AbstractLogger]

class LogCollator:

    def __init__(
        self,
        runid:str,
        root:str,
        # rank:str,
        # local_rank:str,
        loggers:LoggerSequence,
        logfoldername:str='log',
    ):
        self.runid = runid
        self.loggers = loggers
        self.save_folder = os.path.join(root, logfoldername)
        os.makedirs(self.save_folder, exist_ok=True)
        self.file_path = os.path.join(self.save_folder, f"{runid}.jsonl")

    def get_entries(self, **logging_kwargs):
        def t2item(val):
            if torch.is_tensor(val):
                if val.numel() == 1:
                    return val.item()
                return val.tolist()
            return val
        return {
            k:t2item(v) for logger in self.loggers 
            for k, v in logger(**logging_kwargs).items()
        }

    def __call__(self, **logging_kwargs):
        timestamp = logging_kwargs['time']
        log_entry = {
            'time': timestamp, 
            **self.get_entries(**logging_kwargs)
        }
        mode = 'a' if os.path.isfile(self.file_path) else 'w'
        # print(log_entry)
        with open(self.file_path, mode) as log_file:
            log_file.write(json.dumps(log_entry))
            log_file.write('\n')

    @classmethod
    def standard_logger(cls, runid:str, root:str, logfoldername:str='log'):
        loggers = [
            ProgressLogger(), DeltaTimeLogger(), LossLogger(), 
            AccuracyLogger(1), AccuracyLogger(5), LRLogger()
        ]
        return cls(runid, root, loggers, logfoldername)




# class SmoothedValue:

#     def __init__(self, window_size=20, fmt=None):
#         if fmt is None:
#             fmt = "{median:.4f} ({global_avg:.4f})"
#         self.deque = deque(maxlen=window_size)
#         self.total = 0.0
#         self.count = 0
#         self.fmt = fmt

#     def update(self, value, n=1):
#         self.deque.append(value)
#         self.count += n
#         self.total += value * n

#     @property
#     def median(self):
#         d = torch.tensor(list(self.deque))
#         return d.median().item()

#     @property
#     def avg(self):
#         d = torch.tensor(list(self.deque), dtype=torch.float32)
#         return d.mean().item()

#     @property
#     def global_avg(self):
#         return self.total / self.count

#     @property
#     def max(self):
#         return max(self.deque)

#     @property
#     def value(self):
#         return self.deque[-1]

#     def __str__(self):
#         return self.fmt.format(
#             median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
#         )

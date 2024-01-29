import torch
import os
import json
import time
import logging

from typing import Dict, Any, Sequence, Dict

StrDict = Dict[str, Any]

def _getrunlog():
    logger = logging.getLogger('quix.log')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logfmt = logging.Formatter('%(levelname)s:%(asctime)s | %(message)s')
    ch.setFormatter(logfmt)
    logger.addHandler(ch)
    return logger

applog = _getrunlog()

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
        super().__init__([
            'STATUS', 'epoch', 'iteration', 'training', 
            'step_skipped', 'cfg'
        ])


class LossLogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(['loss'])


class LRLogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(['last_lr'])


class GPULogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__([])
    
    @staticmethod
    def getmem(d=None) -> float:
        '''Returns reserved memory on device.

        Args:
            d (torch.device): A torch device.
        
        Returns:
            float: Currently reserved memory on device.
        '''
        if d is not None and d.type == 'cpu':
                return 0
        a, b = torch.cuda.mem_get_info(d)
        return (b-a) / (1024**2)
    
    def log(self, **logging_kwargs):
        return {'gpumem': self.getmem()}


class AccuracyLogger(AbstractLogger):

    def __init__(self, top_k:int, *args, **kwargs):
        super().__init__(['outputs', 'targets'])
        self.logname = f'Acc{top_k}'
        self.top_k = top_k

    def unpack(self, name, dict):
        val = dict.get(name, None)
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
        if outputs is None and targets is None:
            return {}
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
        runid:str, # TODO: We probably don't need this to be required.
        root:str,
        rank:int,
        local_rank:int,
        loggers:LoggerSequence,
        logfoldername:str='log',
        stdout:bool=False,
        drop_debug_entries:Sequence[str]=['time', 'cfg'],
    ):
        self.runid = runid
        self.loggers = loggers
        self.save_folder = os.path.join(root, logfoldername)
        self.file_name = f"{runid}_{rank}_{local_rank}.jsonl"
        self.rank = rank
        self.local_rank = local_rank
        self.stdout = stdout
        self.drop_debug_entries = drop_debug_entries
        os.makedirs(self.save_folder, exist_ok=True)
        self.file_path = os.path.join(self.save_folder, self.file_name)

    @property
    def _use_applog(self) -> bool:
        return self.stdout and self.rank == 0 and self.local_rank == 0

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
        timestamp = logging_kwargs.get('time', time.time())
        log_entry = {
            'time': timestamp, 
            **self.get_entries(**logging_kwargs)
        }
        mode = 'a' if os.path.isfile(self.file_path) else 'w'
        if self._use_applog:
            applog.debug(
                ' '.join([
                    f"{k}={v}" for k,v in log_entry.items() 
                    if k not in self.drop_debug_entries
                ])
            )
        with open(self.file_path, mode) as log_file:
            log_file.write(json.dumps(log_entry))
            log_file.write('\n')

    @classmethod
    def standard_logger(
        cls, runid:str, root:str, rank:int, local_rank:int,
        stdout:bool, logfoldername:str='log'
    ):
        loggers = [
            ProgressLogger(), DeltaTimeLogger(), LossLogger(), 
            AccuracyLogger(1), AccuracyLogger(5), LRLogger(),
            GPULogger()
        ]
        return cls(
            runid, root, rank, local_rank, loggers, 
            logfoldername, stdout
        )

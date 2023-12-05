from typing import Generic
from .dataset import QuixDataset, QuixUnionDataset
from .encoders import (
    EncoderDecoder, CLS, PIL, NPY, RLE, SEG8, SEG16, SEG32, SEG24, DEFAULT_DECODERS, DEFAULT_ENCODERS
)
from .writer import IndexedTarWriter, IndexedShardWriter, QuixWriter
from .aug import parse_train_augs, parse_val_augs
from ..cfg import TDat, TAug

# class QuixDataHandler(Generic[TDat,TAug]):

#     def __init__(self, datcfg:TDat, augcfg:TAug):
#         self.datcfg = datcfg
#         self.augcfg = augcfg
    
#     def load_data(self):
#         pass

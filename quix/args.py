import argparse

# --- Setup Parser Hierarchy ---
# Main parser
parser = argparse.ArgumentParser()

# Tier 1 parsers; model, data, optimizer, logging
modgrp = parser.add_argument_group('model')
datgrp = parser.add_argument_group('data')
optgrp = parser.add_argument_group('optimizer')
loggrp = parser.add_argument_group('logging')

# Tier 2 parsers; augmentations, schedulers
auggrp = datgrp.add_argument_group('augmentations')
schgrp = optgrp.add_argument_group('schedulers')

# --- Add General Arguments ---
# General arguments
parser.add_argument('-e', '--epochs', type=int, default=None, help='No. epoch for training.')
parser.add_argument('-bs', '--batchsize', type=int, default=None, help='Effective batch size.')

# Model arguments, very general, user expandable
modgrp.add_argument('modname', help='Name of model', type=str)

# Data arguments
datgrp.add_argument('dataloc', help='Dataset location.', type=str)
datgrp.add_argument('dataname', help='Dataset name.', type=str)

# Optimizer arguments, no arguments for loss function and loss kwargs
optgrp.add_argument('opt', help='Optimizer name.', type=str)
optgrp.add_argument('-lr', '--learningrate', help='Base / peak learning rate.', type=float, default=None)
optgrp.add_argument('-wd', '--weightdecay', help='Weight decay.', type=float, default=None)
optgrp.add_argument('-eps', '--epsilon', help='Epsilon for adaptive moment.', type=float, default=None)
optgrp.add_argument('-gc', '--gradclip', help='Gradient clipping.', type=float, default=None)
optgrp.add_argument('-as', '--accumulation', type=int, default=None, help='No. batch accumulation steps.')
optgrp.add_argument('--amsgrad', help='Use AMSGrad.', action='store_true', default=False)

# Logging arguments, no expressive set of default loggers
loggrp.add_argument('-lf', '--logfrequency', help='Logging frequency (in iterations).', type=int, default=None)
loggrp.add_argument('-sdir', '--savedir', help='Output directory of logs and checkpoints.', type=str, default=None)
loggrp.add_argument('--stdout', help='Print logs to stdout.', action='store_true', default=False)

# Augmentation arguments, mostly basic stuff, custom augmentations needs to be user specified
auggrp.add_argument('--rrcscale', help='RandomResized', nargs=2, type=int, default=None)
auggrp.add_argument('--rrcratio', help='RandomResized', nargs=2, type=int, default=None)
auggrp.add_argument('--intpmodes', help='Interpolation modes for augmentations', 
    nargs='*', choices=['nearest', 'bilinear', 'bicubic'], default=('all',)
)
auggrp.add_argument('--hflip', help='Use Aug. Hor. Flip.', action='store_true', default=False)
auggrp.add_argument('--vflip', help='Use Aug. Ver. Flip.', action='store_true', default=False)
auggrp.add_argument('--aug3', help='Use aug3 (from DEiTv3).', action='store_true', default=False)
auggrp.add_argument('--randaug', help='RandAug Level.', type=str, default='none', choices=['none', 'light', 'medium', 'strong'])
auggrp.add_argument('--cutmix', help='Use CutMix.', action='store_true', default=False)
auggrp.add_argument('--mixup', help='Use MixUp.', action='store_true', default=False)


if __name__ == '__main__':
    # For testing purposes only
    parser.parse_args()

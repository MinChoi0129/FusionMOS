#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from modules.MFMOS import *
from modules.trainer import Trainer
import __init__ as booger
import numpy as np
import random
from torch import distributed as dist
import torch
from utils.utils import *

DEBUG_MODE = False

print("device count: ", torch.cuda.device_count())

if not DEBUG_MODE:
    dist.init_process_group(backend="nccl")
    print("world_size: ", dist.get_world_size())


# from modules.SalsaNextWithMotionAttention import *


def set_seed(seed=1024):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    # If we need to reproduce the results, increase the training speed
    #    set benchmark = False
    # If we don’t need to reproduce the results, improve the network performance as much as possible
    #    set benchmark = True


if __name__ == "__main__":
    parser = get_args(flags="train")
    FLAGS, unparsed = parser.parse_known_args()

    local_rank = 0
    if not DEBUG_MODE:
        local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    print(f"Local Rank {local_rank}를 설정합니다.")

    FLAGS.log = os.path.join(
        FLAGS.log, datetime.now().strftime("%Y-%-m-%d-%H:%M") + FLAGS.name
    )

    # open arch / data config file
    ARCH = load_yaml(FLAGS.arch_cfg)
    DATA = load_yaml(FLAGS.data_cfg)

    params = MFMOS(nclasses=3, params=ARCH, movable_nclasses=3)
    pytorch_total_params = sum(
        p.numel() for p in params.parameters() if p.requires_grad
    )
    del params

    if local_rank == 0:
        make_logdir(FLAGS=FLAGS, resume_train=False)  # create log folder
        check_pretrained_dir(FLAGS.pretrained)  # does model folder exist?
        backup_to_logdir(FLAGS=FLAGS)  # backup code and config files to logdir

    set_seed()
    # create trainer and start the training
    trainer = Trainer(
        ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.pretrained, local_rank=local_rank
    )

    if local_rank == 0:
        print("----------")
        print("INTERFACE:")
        print("  dataset:", FLAGS.dataset)
        print("  arch_cfg:", FLAGS.arch_cfg)
        print("  data_cfg:", FLAGS.data_cfg)
        print(
            "  Total of Trainable Parameters: {}".format(
                millify(pytorch_total_params, 2)
            )
        )
        print("  log:", FLAGS.log)
        print("  pretrained:", FLAGS.pretrained)
        print(
            "  Augmentation for residual: {}, interval in validation: {}".format(
                ARCH["train"]["residual_aug"], ARCH["train"]["valid_residual_delta_t"]
            )
        )
        print("----------\n")

    trainer.train()

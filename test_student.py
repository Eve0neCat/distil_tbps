from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model.build import build_model, build_student_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs

import torch_npu
from torch_npu.contrib import transfer_to_npu


def calculate_model_size(model):
    model_size = sum(p.numel() for p in model.parameters())
    return model_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/20250124_150702_tinyclip_yfcc15m_xsmall-dist_ours_base_MoTIS_loss2/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    args.training = False
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    # device = "cuda"
    device = args.device
    torch.cuda.set_device(device)

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    # model = build_model(args, num_classes=num_classes)
    model, stu_config = build_student_model(args, convert_fp16=True)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader)
    
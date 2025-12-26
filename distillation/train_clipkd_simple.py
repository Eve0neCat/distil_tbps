import os
import sys
sys.path.append("/home/lugengyou/workspace/Codes/lugy_research/distil_tbps/")

import os.path as op
import numpy as np
import random
import time
import argparse

from model.build import build_student_model, build_teacher_model, Project
from utils.iotools import load_train_configs, save_train_configs
from model import objectives

from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from solver.build import build_optimizer_vision_from_scratch
from model import build_model
from utils.metrics import Evaluator, Evaluator2, Sequence_Distillation_Evaluator, Sequence_Distillation_Evaluator2
from utils.options import get_args
from utils.comm import get_rank, synchronize

from hooks import *
from losses import *
from distillation_processor import do_train_distillation_kd

# Atlas 800(9000) PyTorch 1.11.0
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = get_args()
    set_seed(1 + get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    device = args.device
    torch.cuda.set_device(device)

    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)

    """ get image-text pair datasets dataloader """
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)

    """ build student model """
    student_model, student_config = build_student_model(args)
    logger.info('Student model total params: %2.fM' % (sum(p.numel() for p in student_model.parameters()) / 1000000.0))

    if args.distillation:
        """ build teacher model """
        teacher_config = load_train_configs(args.teacher_model_config)
        teacher_config.training = False
        logger.info(teacher_config)

        teacher_model, teacher_config = build_teacher_model(teacher_config)
        logger.info('Teacher model total params: %2.fM' % (sum(p.numel() for p in teacher_model.parameters()) / 1000000.0))
        tch_checkpointer = Checkpointer(teacher_model, logger=logger)
        tch_checkpointer.load(f=op.join(teacher_config.output_dir, 'best.pth'))

        use_projection_text = student_config.transformer_width != teacher_config.transformer_width
        if use_projection_text and student_config.distillation_config["use_hidden_states"]:
            project_text_hidden_state = Project(student_config.transformer_width, teacher_config.transformer_width)
        if use_projection_text and student_config.distillation_config["use_value_states"]:
            project_text_value_state = Project(student_config.transformer_width, teacher_config.transformer_width)

        use_projection_visual = student_config.vision_width != teacher_config.vision_width
        if use_projection_visual and student_config.distillation_config["use_hidden_states"]:
            project_visual_hidden_state = Project(student_config.vision_width, teacher_config.vision_width)
        if use_projection_visual and student_config.distillation_config["use_value_states"]:
            project_visual_value_state = Project(student_config.vision_width, teacher_config.vision_width)

        distill_config = {"nn_module_names": []}
        distill_hooks_student, distill_hooks_teacher = DistillHooks(distill_config), DistillHooks(distill_config)

        student_model.register_forward_hook(distill_hooks_student.child_to_main_hook)
        teacher_model.register_forward_hook(distill_hooks_teacher.child_to_main_hook)

        project = {}
        if use_projection_text and student_config.distillation_config["use_hidden_states"]:
            project_text_hidden_state.to(device)
            project["project_text_hidden_state"] = project_text_hidden_state
        if use_projection_visual and student_config.distillation_config["use_hidden_states"]:
            project_visual_hidden_state.to(device)
            project["project_visual_hidden_state"] = project_visual_hidden_state
        if use_projection_text and student_config.distillation_config["use_value_states"]:
            project_text_value_state.to(device)
            project["project_text_value_state"] = project_text_value_state
        if use_projection_visual and student_config.distillation_config["use_value_states"]:
            project_visual_value_state.to(device)
            project["project_visual_value_state"] = project_visual_value_state

        teacher_model.to(device)

    student_model.to(device)

    optimizer = build_optimizer(student_config, student_model)
    scheduler = build_lr_scheduler(student_config, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(student_model, optimizer, scheduler, student_config.output_dir, is_master)
    if student_config.continue_training != "":
        checkpointer.load(f=student_config.continue_training)

    if args.dataset_name == "CUHK-PEDES" or args.dataset_name == "RSTPReid":
        evaluator = Sequence_Distillation_Evaluator(val_img_loader, val_txt_loader)
    else:
        evaluator = Sequence_Distillation_Evaluator2(val_img_loader, val_txt_loader)

    if student_config.distillation:
        logger.info("Validation Teacher model:")
        evaluator.eval(teacher_model.eval())

    logger.info("Validation Student model:")
    evaluator.eval(student_model.eval())

    start_epoch = student_config.start_epoch
    if student_config.distillation:
        do_train_distillation_kd(start_epoch, student_model, student_config,
                              train_loader, evaluator, optimizer, scheduler,
                              checkpointer, logger,
                              teacher_model, teacher_config, project)
    else:
        do_train_distillation_kd(start_epoch, student_model, student_config,
                              train_loader, evaluator, optimizer, scheduler,
                              checkpointer, logger)
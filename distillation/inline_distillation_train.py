
import os
import sys
sys.path.append("/home/lugengyou/workspace/Codes/lugy_research/distil_tbps/")

import os.path as op
import numpy as np
import random
import time
import argparse

from model.build import build_student_model, build_teacher_model, Project
from utils.iotools import load_train_configs
from model import objectives

from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from solver.build import build_optimizer_vision_from_scratch
from model import build_model
from utils.metrics import Evaluator, Sequence_Distillation_Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

from hooks import *
from losses import *
from distillation_processor import do_train_distillation

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
    set_seed(1+get_rank())
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
    # for key, value in student_model.named_parameters():
    #     print(key)
    logger.info('Student model total params: %2.fM' % (sum(p.numel() for p in student_model.parameters()) / 1000000.0))
    # student_model.to(device)

    # logger.info(str(student_config).replace(',', '\n'))
    # save_train_configs(student_config.output_dir, student_config)    
    # logger.info('Total params: %2.fM' % (sum(p.numel() for p in student_model.parameters()) / 1000000.0))    

    # print(student_model)
    # image = torch.randn(1, 3, 384, 128).to(device)
    # text = torch.randint(0, 49408, (1, 77)).type(torch.long).to(device)
    # batch = {'images': image, 'caption_ids': text}    
    # student_model(batch)

    if args.distillation:
        """ build teacher model """ 
        teacher_config = load_train_configs(args.teacher_model_config)
        teacher_config.training = True    
        logger.info(teacher_config)

        teacher_model, teacher_config = build_teacher_model(teacher_config)
        logger.info('Teacher model total params: %2.fM' % (sum(p.numel() for p in teacher_model.parameters()) / 1000000.0))
        # checkpointer = Checkpointer(teacher_model)
        # checkpointer.load(f=op.join(teacher_config.output_dir, 'best.pth'))
        # teacher_model.to(device)
        # print(teacher_model)

        # image = torch.randn(1, 3, 384, 128).to(device)
        # text = torch.randint(0, 49408, (1, 77)).type(torch.long).to(device)
        # batch = {'images': image, 'caption_ids': text}    
        # teacher_model(batch)

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

        distill_config = {"nn_module_names": []} #Empty list since we don't want to use nn module hooks here
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
    

    # size = 0
    # for n, p in student_model.named_parameters():
    #     logger.info('n: {}'.format(n))
    #     logger.info('p: {}'.format(p.nelement()))
    #     size += p.nelement()
    # logger.info('Total parameters: {}'.format(size))

    # # Prepare optimizer
    # param_optimizer = list(student_model.named_parameters())
    # if use_projection_text and student_config.distillation_config["use_hidden_states"]:
    #     param_optimizer += list(project_text_hidden_state.named_parameters())
    # if use_projection_text and student_config.distillation_config["use_value_states"]:
    #     param_optimizer += list(project_text_value_state.named_parameters())
    # if use_projection_visual and student_config.distillation_config["use_hidden_states"]:
    #     param_optimizer += list(project_visual_hidden_state.named_parameters())
    # if use_projection_visual and student_config.distillation_config["use_value_states"]:
    #     param_optimizer += list(project_visual_value_state.named_parameters())

    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    # optimizer = FusedAdam(optimizer_grouped_parameters,
    #                           lr=args.learning_rate,
    #                           bias_correction=False)
    # scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion, total_steps=args.max_steps)    
    
    
    # if student_config.vision_from_scratch:
    #     optimizer = build_optimizer_vision_from_scratch(student_config, student_model)
    # else:
    #     optimizer = build_optimizer(student_config, student_model)
    
    stu_optimizer = build_optimizer(student_config, student_model)
    stu_scheduler = build_lr_scheduler(student_config, stu_optimizer)

    tch_optimizer = build_optimizer(teacher_config, teacher_model)
    tch_scheduler = build_lr_scheduler(teacher_config, tch_optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(student_model, stu_optimizer, stu_scheduler, student_config.output_dir, is_master)
    if student_config.continue_training != "":
        checkpointer.load(f=student_config.continue_training)
    evaluator = Evaluator(val_img_loader, val_txt_loader) 

    start_epoch = student_config.start_epoch 
    if student_config.distillation:
        do_train_distillation(start_epoch, student_model, student_config, 
                        train_loader, evaluator, optimizer, scheduler, 
                        checkpointer, logger,
                        teacher_model, teacher_config, project)
    else:
        do_train_distillation(start_epoch, student_model, student_config, 
                        train_loader, evaluator, optimizer, scheduler, 
                        checkpointer, logger)



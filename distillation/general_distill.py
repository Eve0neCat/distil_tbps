
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'distillation'))

import os.path as op
import numpy as np
import random
import time
import argparse

from model.build import build_student_model, build_teacher_model, Project
from utils.iotools import load_train_configs
from model import objectives

from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

from hooks import *
from losses import *

# Atlas 800(9000) PyTorch 1.11.0
import torch
from torch.nn import MSELoss, KLDivLoss, CosineEmbeddingLoss
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
    
    """ get image-text pair datasets dataloader """
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)

    """ build student model """
    student_model, student_config = build_student_model(args)
    # student_model.to(device)

    # logger.info(str(student_config).replace(',', '\n'))
    # save_train_configs(student_config.output_dir, student_config)    
    # logger.info('Total params: %2.fM' % (sum(p.numel() for p in student_model.parameters()) / 1000000.0))    

    # print(student_model)
    # image = torch.randn(1, 3, 384, 128).to(device)
    # text = torch.randint(0, 49408, (1, 77)).type(torch.long).to(device)
    # batch = {'images': image, 'caption_ids': text}    
    # student_model(batch)

    """ build teacher model """ 
    parser = argparse.ArgumentParser(description="Teacher model")
    parser.add_argument("--config_file", default=args.teacher_model_config)
    tmp_args = parser.parse_args()
    teacher_config = load_train_configs(tmp_args.config_file)
    teacher_config.training = False    
    logger.info(teacher_config)

    teacher_model, teacher_config = build_teacher_model(teacher_config)
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

    student_model.to(device)
    teacher_model.to(device)

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
    
    # size = 0
    # for n, p in student_model.named_parameters():
    #     logger.info('n: {}'.format(n))
    #     logger.info('p: {}'.format(p.nelement()))
    #     size += p.nelement()
    # logger.info('Total parameters: {}'.format(size))

    # Prepare optimizer
    param_optimizer = list(student_model.named_parameters())
    if use_projection_text and student_config.distillation_config["use_hidden_states"]:
        param_optimizer += list(project_text_hidden_state.named_parameters())
    if use_projection_text and student_config.distillation_config["use_value_states"]:
        param_optimizer += list(project_text_value_state.named_parameters())
    if use_projection_visual and student_config.distillation_config["use_hidden_states"]:
        param_optimizer += list(project_visual_hidden_state.named_parameters())
    if use_projection_visual and student_config.distillation_config["use_value_states"]:
        param_optimizer += list(project_visual_value_state.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # optimizer = FusedAdam(optimizer_grouped_parameters,
    #                           lr=args.learning_rate,
    #                           bias_correction=False)
    # scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion, total_steps=args.max_steps)    
    
    optimizer = build_optimizer(args, student_model)
    scheduler = build_lr_scheduler(args, optimizer)

    # tr_loss, tr_att_loss, tr_rep_loss, tr_value_loss = 0., 0., 0., 0.
    # student_model.train()

    visual_transformer_losses = VisualTransformerLosses(student_config, teacher_config, device, args)
    text_transformer_losses = TextTransformerLosses(student_config, teacher_config, device, args)    
        

    # for n_iter, batch in enumerate(train_loader):
    if 1:
        batch = next(iter(train_loader))
        batch = {k: v.to(device) for k, v in batch.items()}

        student_model(batch)
        # Gather student states extracted by hooks
        # temp_model = unwrap_ddp(student_model)
        student_reps = flatten_states(student_model.distill_states_dict, "hidden_states")
        student_atts = flatten_states(student_model.distill_states_dict, "attention_scores")        
        student_values = flatten_states(student_model.distill_states_dict, "value_states")
        student_pred_states = flatten_states(student_model.distill_states_dict, "pred_states")
        # student_embeddings = flatten_states(student_model.distill_states_dict, "embedding_states")
        # bsz, attn_heads, seq_len, _  = student_atts[0].shape

        #No gradient for teacher training
        with torch.no_grad():
            teacher_model(batch)
        # Gather teacher states extracted by hooks
        # temp_model = unwrap_ddp(teacher_model)
        teacher_reps = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "hidden_states")]
        teacher_atts = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "attention_scores")]        
        teacher_values = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "value_states")]
        teacher_pred_states = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "pred_states")]
        # teacher_embeddings = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "embedding_states")]

        

        # TODO : 使用 attention_scores 进行蒸馏，需要设置 student 和 teacher 的自注意力头数相同

        rep_visual_loss = 0.
        rep_text_loss = 0.
        att_visual_loss = 0.
        att_text_loss = 0.        
        value_visual_loss = 0.
        value_text_loss = 0.
        pred_visual_loss = 0.
        pred_text_loss = 0.
        task_loss = 0.

        #MiniLM
        if student_config.distillation_config["student_teacher_layer_mapping"] == "last_layer":
            if student_config.distillation_config["use_hidden_states"]:
                # new_teacher_reps = [teacher_reps[-1]]
                # new_student_reps = [student_reps[-1]]
                teacher_layer_num = len(teacher_reps)
                student_layer_num = len(student_reps)
                new_teacher_visual_reps = [teacher_reps[int(teacher_layer_num / 2) - 1]]
                new_student_visual_reps = [student_reps[int(student_layer_num / 2) - 1]]
                new_teacher_text_reps = [teacher_reps[-1]]                
                new_student_text_reps = [student_reps[-1]]

            if student_config.distillation_config["use_attention_scores"]:
                # student_atts = [student_atts[-1]]
                # new_teacher_atts = [teacher_atts[-1]]
                teacher_layer_num = len(teacher_atts)
                student_layer_num = len(student_atts)
                new_teacher_visual_atts = [teacher_atts[int(teacher_layer_num / 2) - 1]]
                student_visual_atts = [student_atts[int(student_layer_num / 2) - 1]]
                new_teacher_text_atts = [teacher_atts[-1]]
                student_text_atts = [student_atts[-1]]                
                
            if student_config.distillation_config["use_value_states"]:
                # student_values = [student_values[-1]]
                # new_teacher_values = [teacher_values[-1]]
                teacher_layer_num = len(teacher_values)
                student_layer_num = len(student_values)
                new_teacher_visual_values = [teacher_values[int(teacher_layer_num / 2) - 1]]
                student_visual_values = [student_values[int(student_layer_num / 2) - 1]]
                new_teacher_text_values = [teacher_values[-1]]  
                student_text_values = [student_values[-1]]                
                          
            if student_config.distillation_config["use_pred_states"]:
                new_teacher_visual_pred = [teacher_pred_states[0]]
                new_student_visual_pred = [student_pred_states[0]]
                new_teacher_text_pred = [teacher_pred_states[-1]]                
                new_student_text_pred = [student_pred_states[-1]]  
                              
        else:
            teacher_layer_num = len(teacher_reps)
            student_layer_num = len(student_reps)

            assert teacher_layer_num % student_layer_num == 0

            layers_per_block = int(teacher_layer_num / student_layer_num)
            visual_layer = int(student_layer_num / 2)

            if student_config.distillation_config["use_hidden_states"]:
                # new_teacher_reps = [teacher_reps[i * layers_per_block + layers_per_block - 1]
                #             for i in range(student_layer_num)]
                # new_student_reps = student_reps
                new_teacher_visual_reps = [teacher_values[i * layers_per_block + layers_per_block - 1]
                                        for i in range(0, visual_layer)]
                new_student_visual_reps = [student_values[i * layers_per_block + layers_per_block - 1]
                                    for i in range(0, visual_layer)]
                new_teacher_text_reps = [teacher_values[i * layers_per_block + layers_per_block - 1]
                                    for i in range(visual_layer, student_layer_num)]
                new_student_text_reps = [student_values[i * layers_per_block + layers_per_block - 1]
                                    for i in range(visual_layer, student_layer_num)]

            if student_config.distillation_config["use_attention_scores"]:
                # new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                #                     for i in range(student_layer_num)]
                new_teacher_visual_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(0, visual_layer)]
                student_visual_atts = [student_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(0, visual_layer)]
                new_teacher_text_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(visual_layer, student_layer_num)]
                student_text_atts = [student_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(visual_layer, student_layer_num)]

            if student_config.distillation_config["use_value_states"]:
                # new_teacher_values = [teacher_values[i * layers_per_block + layers_per_block - 1]
                #             for i in range(student_layer_num)]
                new_teacher_visual_values = [teacher_values[i * layers_per_block + layers_per_block - 1]
                                        for i in range(0, visual_layer)]
                student_visual_values = [student_values[i * layers_per_block + layers_per_block - 1]
                                    for i in range(0, visual_layer)]
                new_teacher_text_values = [teacher_values[i * layers_per_block + layers_per_block - 1]
                                    for i in range(visual_layer, student_layer_num)]
                student_text_values = [student_values[i * layers_per_block + layers_per_block - 1]
                                    for i in range(visual_layer, student_layer_num)]
                
            if student_config.distillation_config["use_pred_states"]:
                new_teacher_visual_pred = [teacher_pred_states[0]]
                new_student_visual_pred = [student_pred_states[0]]
                new_teacher_text_pred = [teacher_pred_states[-1]]                
                new_student_text_pred = [student_pred_states[-1]]

        if student_config.distillation_config["use_hidden_states"]:
            # if use_projection:
            #     rep_loss = transformer_losses.compute_loss(project(new_student_reps), new_teacher_reps, loss_name="hidden_state_loss")
            # else:
            #     rep_loss = transformer_losses.compute_loss(new_student_reps, new_teacher_reps, loss_name="hidden_state_loss")
            if use_projection_visual:
                rep_visual_loss = visual_transformer_losses.compute_loss(project_visual_hidden_state(new_student_visual_reps), new_teacher_visual_reps, loss_name="hidden_state_loss")
            else:
                rep_visual_loss = visual_transformer_losses.compute_loss(new_student_visual_reps, new_teacher_visual_reps, loss_name="hidden_state_loss")

            if use_projection_text:
                rep_text_loss = text_transformer_losses.compute_loss(project_text_hidden_state(new_student_text_reps), new_teacher_text_reps, loss_name="hidden_state_loss")
            else:
                rep_text_loss = text_transformer_losses.compute_loss(new_student_text_reps, new_teacher_text_reps, loss_name="hidden_state_loss")

        if student_config.distillation_config["use_attention_scores"]:
            # att_loss = transformer_losses.compute_loss(student_atts, new_teacher_atts, loss_name="attention_loss")
            att_visual_loss = visual_transformer_losses.compute_loss(student_visual_atts, new_teacher_visual_atts, loss_name="attention_loss")
            att_text_loss = text_transformer_losses.compute_loss(student_text_atts, new_teacher_text_atts, loss_name="attention_loss")
                
        if student_config.distillation_config["use_value_states"]:
            # value_loss = transformer_losses.compute_loss(student_values, new_teacher_values, loss_name="value_state_loss")
            
            # value_visual_loss = visual_transformer_losses.compute_loss(student_visual_values, new_teacher_visual_values, loss_name="value_state_loss")
            # value_text_loss = text_transformer_losses.compute_loss(student_text_values, new_teacher_text_values, loss_name="value_state_loss")
            if use_projection_visual:
                value_visual_loss = visual_transformer_losses.compute_loss(project_visual_value_state(student_visual_values), new_teacher_visual_values, loss_name="value_state_loss")                
            else:
                value_visual_loss = visual_transformer_losses.compute_loss(student_visual_values, new_teacher_visual_values, loss_name="value_state_loss")

            if use_projection_text:
                value_text_loss = text_transformer_losses.compute_loss(project_text_value_state(student_text_values), new_teacher_text_values, loss_name="value_state_loss")
            else:
                value_text_loss = text_transformer_losses.compute_loss(student_text_values, new_teacher_text_values, loss_name="value_state_loss")

        # TODO : 加入模型最后输出层的蒸馏

        if student_config.distillation_config["use_pred_states"]:
            pass

        # 加入 TBPS 任务 sdm loss 损失计算
        if student_config.distillation_config["use_pred_states"] == "sdm":
            logit_scale = torch.ones([]) * (1 / student_config.temperature)
            task_loss = objectives.compute_sdm(new_student_visual_pred[0], new_student_text_pred[0], batch['pids'], logit_scale)

        # loss = att_loss + rep_loss + value_loss
        loss = att_visual_loss + att_text_loss + \
                rep_visual_loss + rep_text_loss + \
                value_visual_loss + value_text_loss + \
                pred_visual_loss + pred_text_loss + \
                task_loss





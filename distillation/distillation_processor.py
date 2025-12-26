import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.checkpoint import Checkpointer
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable

from hooks import flatten_states

from model import objectives
from cross_model_kd_loss import distillation_cross_kd_loss
from kl_loss import distillation_cross_kl_loss, distillation_cross_rkl_loss
from MoTIS_loss import distillation_MoTIS_loss, distillation_reid_MoTIS_loss
from sd_loss import distillation_intra_model_stu_stu_sd_loss, distillation_inter_model_stu_stu_sd_loss
from sd_loss import distillation_intra_model_tch_stu_sd_loss, distillation_inter_model_tch_stu_sd_loss
from sd_loss import distillation_intra_model_tch_stu_sym_sd_loss
from kl_loss import distillation_inter_tch_stu_sym_kl_loss
from kl_loss import distillation_inter_stu_stu_kl_loss, distillation_inter_tch_stu_kl_loss
from sdm_loss import distillation_soft_sdm_loss, compute_inter_tch_stu_sdm_loss
from losses import distillation_fd_loss
from clip_kd_loss import CLIPKDLoss, compute_clip_loss


from solver import build_optimizer, build_lr_scheduler
from collections.abc import Mapping

def get_loss(loss):
    try:
        return loss.item()
    except:
        return 0.0

def tch_stu_loss_weight(epoch, init_alpha=0.9, last_alpha=0.1, start_epoch=10, end_epoch=50):
    if epoch < start_epoch:
        altha = init_alpha
    elif epoch < end_epoch:
        altha = init_alpha - (init_alpha - last_alpha) * (epoch - start_epoch) / (end_epoch - start_epoch)
    else:
        altha = last_alpha
    return altha





def do_train_distillation(start_epoch, student_model, student_config, 
                        train_loader, evaluator, optimizer, scheduler, 
                        checkpointer, logger,
                        teacher_model=None, teacher_config=None, project=None):

    log_period = int(student_config.log_period)
    eval_period = int(student_config.eval_period) 
    # device = "cuda"
    device = next(student_model.parameters()).device
    num_epoch = student_config.num_epoch
    
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    # logger = logging.getLogger("Distillation.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "task_loss": AverageMeter(),
        "distillation_pred_loss": AverageMeter(),
        # "itc_loss": AverageMeter(),
        # "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=student_config.output_dir)

    tchv_stut_best_top1 = 0.0
    tcht_stuv_best_top1 = 0.0
    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        student_model.train()

        for n_iter, batch in enumerate(train_loader):
            
            batch = {k: v.to(device) for k, v in batch.items()}

            # batch_size = batch['images'].shape[0]
            # pid = batch['pids']
            # pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
            # pid_dist = pid - pid.t()
            # labels = (pid_dist == 0).float()
            # print(labels)

            i_feat, t_feat = student_model(batch)

            rep_visual_loss = 0.
            rep_text_loss = 0.
            att_visual_loss = 0.
            att_text_loss = 0.        
            value_visual_loss = 0.
            value_text_loss = 0.
            
            distillation_pred_loss = 0.    
            task_loss = 0.
            
            if student_config.distillation:
                # Gather student states extracted by hooks
                # temp_model = unwrap_ddp(student_model)
                student_reps = flatten_states(student_model.distill_states_dict, "hidden_states")
                student_atts = flatten_states(student_model.distill_states_dict, "attention_scores")        
                student_values = flatten_states(student_model.distill_states_dict, "value_states")
                student_pred_states = flatten_states(student_model.distill_states_dict, "pred_states")
            
                #No gradient for teacher training
                with torch.no_grad():
                    teacher_model(batch)
                # Gather teacher states extracted by hooks
                # temp_model = unwrap_ddp(teacher_model)
                teacher_reps = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "hidden_states")]
                teacher_atts = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "attention_scores")]        
                teacher_values = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "value_states")]
                teacher_pred_states = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "pred_states")]
 

                

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
                        new_teacher_visual_pred = [teacher_pred_states[-1]]
                        new_student_visual_pred = [student_pred_states[-1]]
                        new_teacher_text_pred = [teacher_pred_states[0]]                
                        new_student_text_pred = [student_pred_states[0]]  
                                    
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
                        new_teacher_visual_pred = [teacher_pred_states[-1]]
                        new_student_visual_pred = [student_pred_states[-1]]
                        new_teacher_text_pred = [teacher_pred_states[0]]                
                        new_student_text_pred = [student_pred_states[0]]

                # if student_config.distillation_config["use_hidden_states"]:
                #     # if use_projection:
                #     #     rep_loss = transformer_losses.compute_loss(project(new_student_reps), new_teacher_reps, loss_name="hidden_state_loss")
                #     # else:
                #     #     rep_loss = transformer_losses.compute_loss(new_student_reps, new_teacher_reps, loss_name="hidden_state_loss")
                #     if use_projection_visual:
                #         rep_visual_loss = visual_transformer_losses.compute_loss(project_visual_hidden_state(new_student_visual_reps), new_teacher_visual_reps, loss_name="hidden_state_loss")
                #     else:
                #         rep_visual_loss = visual_transformer_losses.compute_loss(new_student_visual_reps, new_teacher_visual_reps, loss_name="hidden_state_loss")

                #     if use_projection_text:
                #         rep_text_loss = text_transformer_losses.compute_loss(project_text_hidden_state(new_student_text_reps), new_teacher_text_reps, loss_name="hidden_state_loss")
                #     else:
                #         rep_text_loss = text_transformer_losses.compute_loss(new_student_text_reps, new_teacher_text_reps, loss_name="hidden_state_loss")

                # if student_config.distillation_config["use_attention_scores"]:
                #     # att_loss = transformer_losses.compute_loss(student_atts, new_teacher_atts, loss_name="attention_loss")
                #     att_visual_loss = visual_transformer_losses.compute_loss(student_visual_atts, new_teacher_visual_atts, loss_name="attention_loss")
                #     att_text_loss = text_transformer_losses.compute_loss(student_text_atts, new_teacher_text_atts, loss_name="attention_loss")
                        
                # if student_config.distillation_config["use_value_states"]:
                #     # value_loss = transformer_losses.compute_loss(student_values, new_teacher_values, loss_name="value_state_loss")
                    
                #     # value_visual_loss = visual_transformer_losses.compute_loss(student_visual_values, new_teacher_visual_values, loss_name="value_state_loss")
                #     # value_text_loss = text_transformer_losses.compute_loss(student_text_values, new_teacher_text_values, loss_name="value_state_loss")
                #     if use_projection_visual:
                #         value_visual_loss = visual_transformer_losses.compute_loss(project_visual_value_state(student_visual_values), new_teacher_visual_values, loss_name="value_state_loss")                
                #     else:
                #         value_visual_loss = visual_transformer_losses.compute_loss(student_visual_values, new_teacher_visual_values, loss_name="value_state_loss")

                #     if use_projection_text:
                #         value_text_loss = text_transformer_losses.compute_loss(project_text_value_state(student_text_values), new_teacher_text_values, loss_name="value_state_loss")
                #     else:
                #         value_text_loss = text_transformer_losses.compute_loss(student_text_values, new_teacher_text_values, loss_name="value_state_loss")

                # 模型最后输出层的知识蒸馏
                if student_config.distillation_config["use_pred_states"]:
                    # cross_kl loss: Hinton KD: 使用正向 KL 散度，计算两个模型之间的相似性
                    if "cross_kl_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_cross_kl_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config)
                    # cross_kl loss: Hinton KD: 使用反向 KL 散度，计算两个模型之间的相似性
                    if "cross_rkl_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_cross_rkl_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config)
                    # cross_kd loss: Hinton KD: 使用教师模型的预测结果作为软标签，进行知识蒸馏
                    if "cross_kd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_cross_kd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config)

                    if "ssdm_loss" in student_config.distillation_config["pred_states_loss"]: 
                        distillation_pred_loss += distillation_soft_sdm_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config, batch['pids'])
                    # MoTIS loss
                    if "MoTIS" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_MoTIS_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config, vision_distill=True, text_distill=True)

                    if "inter_ss_kl_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_stu_stu_kl_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config)

                    # sd loss
                    if "intra_ss_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_intra_model_stu_stu_sd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0])
                    if "inter_ss_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_model_stu_stu_sd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0])
                    if "intra_ts_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_intra_model_tch_stu_sd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0])
                    if "inter_ts_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_model_tch_stu_sd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0])
                    if "intra_ts_sym_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_intra_model_tch_stu_sym_sd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0])              

                    # sym-kl-div
                    if "inter_ts_sym_kl_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_tch_stu_sym_kl_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0], student_config) 
                        
                    
                    if "fd" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_fd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0])
                   
                    if "reid_MoTIS_loss" in student_config.distillation_config["pred_states_loss"]:
                        vision_distill = student_config.reid_MoTIS_vision
                        text_distill = student_config.reid_MoTIS_text
                        distillation_pred_loss += distillation_reid_MoTIS_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                            new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                            student_config, batch['pids'],
                                                                            vision_distill=vision_distill, text_distill=text_distill)

                # 加入 TBPS 任务 sdm loss 损失计算
                if student_config.distillation_config["use_task_loss"]:  
                    # 学生图像编码器与教师文本编码器之间的 任务损失
                    if "st_itc" in student_config.distillation_config["task_loss"]:
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        # 共享教师模型的文本编码器
                        task_loss += objectives.compute_itc(new_student_visual_pred[0], new_teacher_text_pred[0], logit_scale) 
                    if "st_sdm" in student_config.distillation_config["task_loss"]:                                         
                        # 共享教师模型的文本编码器，进计算视觉编码器的 sdm loss
                        task_loss += compute_inter_tch_stu_sdm_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                new_teacher_visual_pred[0],  new_teacher_text_pred[0], 
                                                                student_config, batch['pids'], vision_loss=True, text_loss=False)
                    # 学生文本编码器与教师图像编码器之间的 任务损失
                    if "ts_itc" in student_config.distillation_config["task_loss"]:
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        task_loss += objectives.compute_itc(new_teacher_visual_pred[0], new_student_text_pred[0], logit_scale)
                    if "ts_sdm" in student_config.distillation_config["task_loss"]:                                         
                        # 共享教师模型的图像编码器，进计算文本编码器的 sdm loss
                        task_loss += compute_inter_tch_stu_sdm_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                new_teacher_visual_pred[0],  new_teacher_text_pred[0], 
                                                                student_config, batch['pids'], vision_loss=False, text_loss=True)  
                    # 学生图像和文本编码器之间的 任务损失
                    if "ss_itc" in student_config.distillation_config["task_loss"]:
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        task_loss += objectives.compute_itc(new_student_visual_pred[0], new_student_text_pred[0], logit_scale) 
                    if "ss_sdm" in student_config.distillation_config["task_loss"]: 
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        task_loss = objectives.compute_sdm(new_student_visual_pred[0], new_student_text_pred[0], batch['pids'], logit_scale) 

            else:
                if "itc" in student_config.loss_names:
                    # itc loss
                    logit_scale = torch.ones([]) * (1 / student_config.temperature)
                    task_loss += objectives.compute_itc(i_feat, t_feat, logit_scale)
                if "sdm" in student_config.loss_names:
                    # sdm loss
                    logit_scale = torch.ones([]) * (1 / student_config.temperature)
                    task_loss += objectives.compute_sdm(i_feat, t_feat, batch['pids'], logit_scale)
                if "smse" in student_config.loss_names:
                    # sd loss
                    task_loss += objectives.compute_self_mse(i_feat, t_feat, batch['pids'], image_mse=True, text_mse=True, altha=0.8)

            if student_config.distillation:
                if student_config.distillation_cosin_weight:
                    # loss weight 计算
                    altha = tch_stu_loss_weight(epoch, init_alpha=0.9, last_alpha=0.5, start_epoch=20, end_epoch=50)
                elif student_config.distillation_constant_weight:
                    altha = student_config.distillation_weight
                else:                    
                    # loss weight 计算            
                    if epoch < 10:
                        altha = 0.9
                    elif start_epoch < 20:
                        altha = 0.6
                    elif start_epoch < 40:
                        altha = 0.3
                    else:
                        altha = 0.1
            else:
                altha = 0.0

            total_loss = att_visual_loss + att_text_loss + \
                        rep_visual_loss + rep_text_loss + \
                        value_visual_loss + value_text_loss + \
                        altha * distillation_pred_loss + (1 - altha) * task_loss
        
            batch_size = batch['images'].shape[0]
            meters['loss'].update(get_loss(total_loss), batch_size)
            meters['task_loss'].update(get_loss(task_loss), batch_size)
            meters['distillation_pred_loss'].update(get_loss(distillation_pred_loss), batch_size)
            # meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            # meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            # meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            # meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            # meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            # meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.6f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        # 绘制到 TensorBoard 进行可视化
        # tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        # tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        # for k, v in meters.items():
        #     if v.avg > 0:
        #         tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Share teacher vision encoder for evaluation ...")
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if teacher_model is not None:
                    top1 = evaluator.eval(student_model.eval(), teacher_model.eval(), use_tch_text=False, use_tch_image=True)
                else:
                    top1 = evaluator.eval(student_model.eval())
                if tchv_stut_best_top1 < top1:
                    tchv_stut_best_top1 = top1
                    arguments["tchv_stut_best_top1_epoch"] = epoch
                    
                logger.info("Share teacher text encoder for evaluation ...")
                logger.info("Validation Results - Epoch: {}".format(epoch))
                top1 = evaluator.eval(student_model.eval(), teacher_model.eval(), use_tch_text=True, use_tch_image=False)
                if tcht_stuv_best_top1 < top1:
                    tcht_stuv_best_top1 = top1
                    arguments["tcht_stuv_best_top1_epoch"] = epoch
                
                # logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("Distillation student model evaluation ...")
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if student_config.distributed:
                    top1 = evaluator.eval(student_model.module.eval())
                else:
                    top1 = evaluator.eval(student_model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())




def do_train_sequence_distillation_MoTIS_3s(start_epoch, student_model, student_config, 
                                        train_loader, evaluator, optimizer, scheduler, 
                                        checkpointer, logger,
                                        teacher_model=None, teacher_config=None, project=None):
    log_period = int(student_config.log_period)
    eval_period = int(student_config.eval_period)   
    device = next(student_model.parameters()).device
    num_epoch = student_config.num_epoch
    
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    # logger = logging.getLogger("Distillation.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "task_loss": AverageMeter(),
        "distillation_pred_loss": AverageMeter(),
        # "itc_loss": AverageMeter(),
        # "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=student_config.output_dir)


    """ fixed student text encoder and only distillate student vision model """
    optimizer = build_optimizer(student_config, student_model)
    scheduler = build_lr_scheduler(student_config, optimizer)
    st_best_top1 = 0.0        
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        student_model.train()

        for n_iter, batch in enumerate(train_loader):
            
            batch = {k: v.to(device) for k, v in batch.items()}

            i_feat, t_feat = student_model(batch)

            rep_visual_loss = 0.
            rep_text_loss = 0.
            att_visual_loss = 0.
            att_text_loss = 0.        
            value_visual_loss = 0.
            value_text_loss = 0.
            
            distillation_pred_loss = 0.    
            task_loss = 0.
            
            if student_config.distillation:
                # Gather student states extracted by hooks
                # temp_model = unwrap_ddp(student_model)
                student_reps = flatten_states(student_model.distill_states_dict, "hidden_states")
                student_atts = flatten_states(student_model.distill_states_dict, "attention_scores")        
                student_values = flatten_states(student_model.distill_states_dict, "value_states")
                student_pred_states = flatten_states(student_model.distill_states_dict, "pred_states")
            
                #No gradient for teacher training
                with torch.no_grad():
                    teacher_model(batch)
                # Gather teacher states extracted by hooks
                # temp_model = unwrap_ddp(teacher_model)
                teacher_reps = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "hidden_states")]
                teacher_atts = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "attention_scores")]        
                teacher_values = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "value_states")]
                teacher_pred_states = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "pred_states")]
 
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
                        new_teacher_visual_pred = [teacher_pred_states[-1]]
                        new_student_visual_pred = [student_pred_states[-1]]
                        new_teacher_text_pred = [teacher_pred_states[0]]                
                        new_student_text_pred = [student_pred_states[0]]  
                                    
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
                        new_teacher_visual_pred = [teacher_pred_states[-1]]
                        new_student_visual_pred = [student_pred_states[-1]]
                        new_teacher_text_pred = [teacher_pred_states[0]]                
                        new_student_text_pred = [student_pred_states[0]]
                
                # 模型最后输出层的知识蒸馏
                if student_config.distillation_config["use_pred_states"]:
                    # 只蒸馏图像编码器
                    if "MoTIS" in student_config.distillation_config["pred_states_loss"]:                        
                        distillation_pred_loss += distillation_MoTIS_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config, vision_distill=True, text_distill=False) 
                    if "inter_ts_kl_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_tch_stu_kl_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config, vision_distill=True, text_distill=False)
                # 加入 TBPS 任务 sdm loss 损失计算
                if student_config.distillation_config["use_task_loss"]: 
                    if "itc" in student_config.distillation_config["task_loss"]:
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        # 共享教师模型的文本编码器
                        task_loss += objectives.compute_itc(new_student_visual_pred[0], new_teacher_text_pred[0], logit_scale)                                                             
                    if "inter_ts_sdm_loss" in student_config.distillation_config["task_loss"]:                                         
                        # 共享教师模型的文本编码器，进计算视觉编码器的 sdm loss
                        task_loss += compute_inter_tch_stu_sdm_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                new_teacher_visual_pred[0],  new_teacher_text_pred[0], 
                                                                student_config, batch['pids'], vision_loss=True, text_loss=False)
            else:
                if "sdm" in student_config.loss_names:
                    # sdm loss
                    logit_scale = torch.ones([]) * (1 / student_config.temperature)
                    task_loss += objectives.compute_sdm(i_feat, t_feat, batch['pids'], logit_scale)

            if student_config.distillation:
                if student_config.distillation_cosin_weight:
                    # loss weight 计算
                    alpha = tch_stu_loss_weight(epoch, init_alpha=0.9, last_alpha=0.5, start_epoch=20, end_epoch=50)
                elif student_config.distillation_constant_weight:
                    alpha = student_config.distillation_weight
                else:                    
                    # loss weight 计算            
                    if epoch < 10:
                        alpha = 0.9
                    elif start_epoch < 20:
                        alpha = 0.6
                    elif start_epoch < 40:
                        alpha = 0.3
                    else:
                        alpha = 0.1
            else:
                alpha = 0.0

            # loss 计算
            total_loss = att_visual_loss + att_text_loss + \
                        rep_visual_loss + rep_text_loss + \
                        value_visual_loss + value_text_loss + \
                        alpha * distillation_pred_loss + (1 - alpha) * task_loss
        
            batch_size = batch['images'].shape[0]
            meters['loss'].update(get_loss(total_loss), batch_size)
            meters['task_loss'].update(get_loss(task_loss), batch_size)
            meters['distillation_pred_loss'].update(get_loss(distillation_pred_loss), batch_size)
            # meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            # meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            # meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            # meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            # meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            # meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.6f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
       
        # 绘制到 TensorBoard 进行可视化
        # tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        # tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        # for k, v in meters.items():
        #     if v.avg > 0:
        #         tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
                       
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Share teacher text encoder for evaluation ...")
                logger.info("Validation Results - Epoch: {}".format(epoch))
                  
                top1 = evaluator.eval(student_model.eval(), teacher_model.eval(), use_tch_text=True, use_tch_image=False)

                torch.cuda.empty_cache()
                if st_best_top1 < top1:
                    st_best_top1 = top1
                    arguments["st_best_top1_epoch"] = epoch
                #     checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"st_best_top1 R1: {st_best_top1} at epoch {arguments['st_best_top1_epoch']}")


    # 重新设置学习率调整器
    optimizer = build_optimizer(student_config, student_model)
    scheduler = build_lr_scheduler(student_config, optimizer)
    ts_best_top1 = 0.0
    ss_best_top1 = 0.0
    is_master = get_rank() == 0
    checkpointer = Checkpointer(student_model, optimizer, scheduler, student_config.output_dir, is_master)
    """ fixed student vision encoder and only ditillate student text model """
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        student_model.train()

        for n_iter, batch in enumerate(train_loader):
            
            batch = {k: v.to(device) for k, v in batch.items()}

            # i_feat, t_feat = student_model(batch)
            # 加入 MoTIS 算法消融实验
            i_feat, t_feat = student_model(batch, vision_grad=False, text_grad=True)
            # i_feat = student_model.encode_image(batch['images']) 
            # with torch.no_grad(): 
            #     # fix student text encoder   
            #     t_feat = student_model.encode_text(batch['caption_ids'])          

            rep_visual_loss = 0.
            rep_text_loss = 0.
            att_visual_loss = 0.
            att_text_loss = 0.        
            value_visual_loss = 0.
            value_text_loss = 0.
            
            distillation_pred_loss = 0.    
            task_loss = 0.
            
            if student_config.distillation:
                # Gather student states extracted by hooks
                # temp_model = unwrap_ddp(student_model)
                student_reps = flatten_states(student_model.distill_states_dict, "hidden_states")
                student_atts = flatten_states(student_model.distill_states_dict, "attention_scores")        
                student_values = flatten_states(student_model.distill_states_dict, "value_states")
                student_pred_states = flatten_states(student_model.distill_states_dict, "pred_states")
            
                #No gradient for teacher training
                with torch.no_grad():
                    teacher_model(batch)
                # Gather teacher states extracted by hooks
                # temp_model = unwrap_ddp(teacher_model)
                teacher_reps = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "hidden_states")]
                teacher_atts = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "attention_scores")]        
                teacher_values = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "value_states")]
                teacher_pred_states = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "pred_states")]
 
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
                        new_teacher_visual_pred = [teacher_pred_states[-1]]
                        new_student_visual_pred = [student_pred_states[-1]]
                        new_teacher_text_pred = [teacher_pred_states[0]]                
                        new_student_text_pred = [student_pred_states[0]]  
                                    
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
                        new_teacher_visual_pred = [teacher_pred_states[-1]]
                        new_student_visual_pred = [student_pred_states[-1]]
                        new_teacher_text_pred = [teacher_pred_states[0]]                
                        new_student_text_pred = [student_pred_states[0]]
                
                # 模型最后输出层的知识蒸馏
                if student_config.distillation_config["use_pred_states"]:
                    # 只蒸馏文本编码器
                    if "MoTIS" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_MoTIS_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config, vision_distill=False, text_distill=True)
                    if "inter_ts_kl_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_tch_stu_kl_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config, vision_distill=False, text_distill=True)
                    # # sym-kl-div
                    # if "inter_ts_sym_kl_loss" in student_config.distillation_config["pred_states_loss"]:
                    #     distillation_pred_loss += distillation_inter_tch_stu_sym_kl_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                    #                                                     new_teacher_visual_pred[0],  new_teacher_text_pred[0], student_config) 

                # 加入 TBPS 任务损失计算
                if student_config.distillation_config["use_task_loss"]: 
                    # 共享教师模型的图像编码器 
                    if "itc" in student_config.distillation_config["task_loss"]:
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        task_loss += objectives.compute_itc(new_teacher_visual_pred[0], new_student_text_pred[0], logit_scale) 
                    # 学生图像和文本编码器联合计算损失
                    if "ss_itc_s2" in student_config.distillation_config["task_loss"]:
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        task_loss += objectives.compute_itc(new_student_visual_pred[0], new_student_text_pred[0], logit_scale)  
            else:
                # sdm loss
                logit_scale = torch.ones([]) * (1 / student_config.temperature)
                task_loss = objectives.compute_sdm(i_feat, t_feat, batch['pids'], logit_scale)

            if student_config.distillation:
                if student_config.distillation_cosin_weight:
                    # loss weight 计算
                    alpha = tch_stu_loss_weight(epoch, init_alpha=0.9, last_alpha=0.5, start_epoch=20, end_epoch=50)
                elif student_config.distillation_constant_weight:
                    alpha = student_config.distillation_weight
                else:                    
                    # loss weight 计算            
                    if epoch < 10:
                        alpha = 0.9
                    elif start_epoch < 20:
                        alpha = 0.6
                    elif start_epoch < 40:
                        alpha = 0.3
                    else:
                        alpha = 0.1
            else:
                alpha = 0.0

            # loss 计算
            total_loss = att_visual_loss + att_text_loss + \
                        rep_visual_loss + rep_text_loss + \
                        value_visual_loss + value_text_loss + \
                        alpha * distillation_pred_loss + (1 - alpha) * task_loss
        
            batch_size = batch['images'].shape[0]
            meters['loss'].update(get_loss(total_loss), batch_size)
            meters['task_loss'].update(get_loss(task_loss), batch_size)
            meters['distillation_pred_loss'].update(get_loss(distillation_pred_loss), batch_size)
            # meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            # meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            # meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            # meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            # meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            # meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.6f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        # 绘制到 TensorBoard 进行可视化
        # tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        # tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        # for k, v in meters.items():
        #     if v.avg > 0:
        #         tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Share teacher vision encoder for evaluation ...")
                logger.info("Validation Results - Epoch: {}".format(epoch))
                top1 = evaluator.eval(student_model.eval(), teacher_model.eval(), use_tch_text=False, use_tch_image=True)
                if ts_best_top1 < top1:
                    ts_best_top1 = top1
                    arguments["ts_best_top1_epoch"] = epoch
                    checkpointer.save("stut_tchv_best", **arguments)  

                logger.info("Distillation student model evaluation ...")
                logger.info("Validation Results - Epoch: {}".format(epoch))
                top1 = evaluator.eval(student_model.eval())

                torch.cuda.empty_cache()
                if ss_best_top1 < top1:
                    ss_best_top1 = top1
                    arguments["ss_best_top1_epoch"] = epoch
                    # checkpointer.save("best", **arguments)                    
    if get_rank() == 0:
        logger.info(f"ts_best_top1 R1: {ts_best_top1} at epoch {arguments['ts_best_top1_epoch']}")
        logger.info(f"ss_best_top1 R1: {ss_best_top1} at epoch {arguments['ss_best_top1_epoch']}")

    
    # 重新设置学习率调整器
    optimizer = build_optimizer(student_config, student_model)
    scheduler = build_lr_scheduler(student_config, optimizer)
    best_top1 = 0.0
    tchv_stut_best_top1 = 0.0
    tcht_stuv_best_top1 = 0.0
    is_master = get_rank() == 0
    checkpointer = Checkpointer(student_model, optimizer, scheduler, student_config.output_dir, is_master)
    """ ditillate student model """
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        student_model.train()

        for n_iter, batch in enumerate(train_loader):
            
            batch = {k: v.to(device) for k, v in batch.items()}

            i_feat, t_feat = student_model(batch)
            # i_feat = student_model.encode_image(batch['images']) 
            # with torch.no_grad(): 
            #     # fix student text encoder   
            #     t_feat = student_model.encode_text(batch['caption_ids'])          

            rep_visual_loss = 0.
            rep_text_loss = 0.
            att_visual_loss = 0.
            att_text_loss = 0.        
            value_visual_loss = 0.
            value_text_loss = 0.
            
            distillation_pred_loss = 0.    
            task_loss = 0.
            
            if student_config.distillation:
                # Gather student states extracted by hooks
                # temp_model = unwrap_ddp(student_model)
                student_reps = flatten_states(student_model.distill_states_dict, "hidden_states")
                student_atts = flatten_states(student_model.distill_states_dict, "attention_scores")        
                student_values = flatten_states(student_model.distill_states_dict, "value_states")
                student_pred_states = flatten_states(student_model.distill_states_dict, "pred_states")
            
                #No gradient for teacher training
                with torch.no_grad():
                    teacher_model(batch)
                # Gather teacher states extracted by hooks
                # temp_model = unwrap_ddp(teacher_model)
                teacher_reps = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "hidden_states")]
                teacher_atts = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "attention_scores")]        
                teacher_values = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "value_states")]
                teacher_pred_states = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "pred_states")]
 
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
                        new_teacher_visual_pred = [teacher_pred_states[-1]]
                        new_student_visual_pred = [student_pred_states[-1]]
                        new_teacher_text_pred = [teacher_pred_states[0]]                
                        new_student_text_pred = [student_pred_states[0]]  
                                    
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
                        new_teacher_visual_pred = [teacher_pred_states[-1]]
                        new_student_visual_pred = [student_pred_states[-1]]
                        new_teacher_text_pred = [teacher_pred_states[0]]                
                        new_student_text_pred = [student_pred_states[0]]
                
                # 模型最后输出层的知识蒸馏
                if student_config.distillation_config["use_pred_states"]:
                    # 联合蒸馏
                    if "MoTIS" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_MoTIS_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config, vision_distill=True, text_distill=True)
                    if "inter_ss_kl_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_stu_stu_kl_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config)
                    # sym-kl-div
                    if "inter_ts_sym_kl_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_tch_stu_sym_kl_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0], student_config) 

                # 加入 TBPS 任务损失计算
                if student_config.distillation_config["use_task_loss"]:                     
                    if "ss_itc" in student_config.distillation_config["task_loss"]:
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        task_loss += objectives.compute_itc(new_student_visual_pred[0], new_student_text_pred[0], logit_scale) 
                    # 共享教师模型的图像编码器 
                    if "ts_itc" in student_config.distillation_config["task_loss"]:
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        task_loss += objectives.compute_itc(new_teacher_visual_pred[0], new_student_text_pred[0], logit_scale)
                    # 共享教师模型的文本编码器 
                    if "st_itc" in student_config.distillation_config["task_loss"]:
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        task_loss += objectives.compute_itc(new_student_visual_pred[0], new_teacher_text_pred[0], logit_scale)                                      
            else:
                # sdm loss
                logit_scale = torch.ones([]) * (1 / student_config.temperature)
                task_loss = objectives.compute_sdm(i_feat, t_feat, batch['pids'], logit_scale)

            if student_config.distillation:                
                if student_config.distillation_cosin_weight:
                    # loss weight 计算
                    alpha = tch_stu_loss_weight(epoch, init_alpha=0.9, last_alpha=0.5, start_epoch=20, end_epoch=50)
                elif student_config.distillation_constant_weight:
                    alpha = student_config.distillation_weight
                else:                    
                    # loss weight 计算            
                    if epoch < 10:
                        alpha = 0.9
                    elif start_epoch < 20:
                        alpha = 0.6
                    elif start_epoch < 40:
                        alpha = 0.3
                    else:
                        alpha = 0.1
            else:
                alpha = 0.0

            # loss 计算
            total_loss = att_visual_loss + att_text_loss + \
                        rep_visual_loss + rep_text_loss + \
                        value_visual_loss + value_text_loss + \
                        alpha * distillation_pred_loss + (1 - alpha) * task_loss
        
            batch_size = batch['images'].shape[0]
            meters['loss'].update(get_loss(total_loss), batch_size)
            meters['task_loss'].update(get_loss(task_loss), batch_size)
            meters['distillation_pred_loss'].update(get_loss(distillation_pred_loss), batch_size)
            # meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            # meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            # meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            # meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            # meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            # meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.6f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        # 绘制到 TensorBoard 进行可视化
        # tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        # tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        # for k, v in meters.items():
        #     if v.avg > 0:
        #         tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Share teacher vision encoder for evaluation ...")
                logger.info("Validation Results - Epoch: {}".format(epoch))
                top1 = evaluator.eval(student_model.eval(), teacher_model.eval(), use_tch_text=False, use_tch_image=True)
                if tchv_stut_best_top1 < top1:
                    tchv_stut_best_top1 = top1
                    arguments["tchv_stut_best_top1_epoch"] = epoch
                    
                logger.info("Share teacher text encoder for evaluation ...")
                logger.info("Validation Results - Epoch: {}".format(epoch))
                top1 = evaluator.eval(student_model.eval(), teacher_model.eval(), use_tch_text=True, use_tch_image=False)
                if tcht_stuv_best_top1 < top1:
                    tcht_stuv_best_top1 = top1
                    arguments["tcht_stuv_best_top1_epoch"] = epoch
                
                logger.info("Distillation student model evaluation ...")
                logger.info("Validation Results - Epoch: {}".format(epoch))
                top1 = evaluator.eval(student_model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"tchv_stut_best_top1 R1: {tchv_stut_best_top1} at epoch {arguments['tchv_stut_best_top1_epoch']}")
        logger.info(f"tcht_stuv_best_top1 R1: {tcht_stuv_best_top1} at epoch {arguments['tcht_stuv_best_top1_epoch']}")
        logger.info(f"best_top1 R1: {best_top1} at epoch {arguments['epoch']}")


def do_train_sequence_distillation_MoTIS_2s(start_epoch, student_model, student_config, 
                                        train_loader, evaluator, optimizer, scheduler, 
                                        checkpointer, logger,
                                        teacher_model=None, teacher_config=None, project=None):
    log_period = int(student_config.log_period)
    eval_period = int(student_config.eval_period)    
    device = next(student_model.parameters()).device
    num_epoch = student_config.num_epoch
    
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    # logger = logging.getLogger("Distillation.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "task_loss": AverageMeter(),
        "distillation_pred_loss": AverageMeter(),
        # "itc_loss": AverageMeter(),
        # "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=student_config.output_dir)


    """ fixed student text encoder and only distillate student vision model """
    optimizer = build_optimizer(student_config, student_model)
    scheduler = build_lr_scheduler(student_config, optimizer)
    st_best_top1 = 0.0        
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        student_model.train()

        for n_iter, batch in enumerate(train_loader):
            
            batch = {k: v.to(device) for k, v in batch.items()}

            i_feat, t_feat = student_model(batch)

            rep_visual_loss = 0.
            rep_text_loss = 0.
            att_visual_loss = 0.
            att_text_loss = 0.        
            value_visual_loss = 0.
            value_text_loss = 0.
            
            distillation_pred_loss = 0.    
            task_loss = 0.
            
            if student_config.distillation:
                # Gather student states extracted by hooks
                # temp_model = unwrap_ddp(student_model)
                student_reps = flatten_states(student_model.distill_states_dict, "hidden_states")
                student_atts = flatten_states(student_model.distill_states_dict, "attention_scores")        
                student_values = flatten_states(student_model.distill_states_dict, "value_states")
                student_pred_states = flatten_states(student_model.distill_states_dict, "pred_states")
            
                #No gradient for teacher training
                with torch.no_grad():
                    teacher_model(batch)
                # Gather teacher states extracted by hooks
                # temp_model = unwrap_ddp(teacher_model)
                teacher_reps = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "hidden_states")]
                teacher_atts = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "attention_scores")]        
                teacher_values = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "value_states")]
                teacher_pred_states = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "pred_states")]
 
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
                        new_teacher_visual_pred = [teacher_pred_states[-1]]
                        new_student_visual_pred = [student_pred_states[-1]]
                        new_teacher_text_pred = [teacher_pred_states[0]]                
                        new_student_text_pred = [student_pred_states[0]]  
                                    
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
                        new_teacher_visual_pred = [teacher_pred_states[-1]]
                        new_student_visual_pred = [student_pred_states[-1]]
                        new_teacher_text_pred = [teacher_pred_states[0]]                
                        new_student_text_pred = [student_pred_states[0]]
        
                # 模型最后输出层的知识蒸馏
                if student_config.distillation_config["use_pred_states"]:
                    # 只蒸馏图像编码器，MoTIS 算法
                    if "MoTIS" in student_config.distillation_config["pred_states_loss"]:                        
                        distillation_pred_loss += distillation_MoTIS_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config, vision_distill=True, text_distill=False) 
                    if "inter_ts_kl_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_tch_stu_kl_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config, vision_distill=True, text_distill=False)
                    # ConaCLIP 算法
                    if "intra_ss_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_intra_model_stu_stu_sd_loss(new_student_visual_pred[0], new_student_text_pred[0],
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0], 
                                                                        vision_distill=True, text_distill=False)
                    if "intra_ts_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_intra_model_tch_stu_sd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        vision_distill=True, text_distill=False)
                    
                # 加入 TBPS 任务 sdm loss 损失计算
                if student_config.distillation_config["use_task_loss"]: 
                    if "st_itc" in student_config.distillation_config["task_loss"]:
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        # 共享教师模型的文本编码器
                        task_loss += objectives.compute_itc(new_student_visual_pred[0], new_teacher_text_pred[0], logit_scale) 
                    if "st_sdm" in student_config.distillation_config["task_loss"]:                                         
                        # 共享教师模型的文本编码器，进计算视觉编码器的 sdm loss
                        task_loss += compute_inter_tch_stu_sdm_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                new_teacher_visual_pred[0],  new_teacher_text_pred[0], 
                                                                student_config, batch['pids'], vision_loss=True, text_loss=False)
            else:
                if "sdm" in student_config.loss_names:
                    # sdm loss
                    logit_scale = torch.ones([]) * (1 / student_config.temperature)
                    task_loss += objectives.compute_sdm(i_feat, t_feat, batch['pids'], logit_scale)

            if student_config.distillation:
                if student_config.distillation_cosin_weight:
                    # loss weight 计算
                    alpha = tch_stu_loss_weight(epoch, init_alpha=0.9, last_alpha=0.5, start_epoch=20, end_epoch=50)
                elif student_config.distillation_constant_weight:
                    alpha = student_config.distillation_weight
                else:                    
                    # loss weight 计算            
                    if epoch < 10:
                        alpha = 0.9
                    elif start_epoch < 20:
                        alpha = 0.6
                    elif start_epoch < 40:
                        alpha = 0.3
                    else:
                        alpha = 0.1
            else:
                alpha = 0.0

            # loss 计算
            total_loss = att_visual_loss + att_text_loss + \
                        rep_visual_loss + rep_text_loss + \
                        value_visual_loss + value_text_loss + \
                        alpha * distillation_pred_loss + (1 - alpha) * task_loss
        
            batch_size = batch['images'].shape[0]
            meters['loss'].update(get_loss(total_loss), batch_size)
            meters['task_loss'].update(get_loss(task_loss), batch_size)
            meters['distillation_pred_loss'].update(get_loss(distillation_pred_loss), batch_size)
            # meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            # meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            # meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            # meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            # meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            # meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.6f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
       
        # 绘制到 TensorBoard 进行可视化
        # tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        # tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        # for k, v in meters.items():
        #     if v.avg > 0:
        #         tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
                       
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Share teacher text encoder for evaluation ...")
                logger.info("Validation Results - Epoch: {}".format(epoch))
                  
                top1 = evaluator.eval(student_model.eval(), teacher_model.eval(), use_tch_text=True, use_tch_image=False)

                torch.cuda.empty_cache()
                if st_best_top1 < top1:
                    st_best_top1 = top1
                    arguments["st_best_top1_epoch"] = epoch
                #     checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"st_best_top1 R1: {st_best_top1} at epoch {arguments['st_best_top1_epoch']}")


    # 重新设置学习率调整器
    optimizer = build_optimizer(student_config, student_model)
    scheduler = build_lr_scheduler(student_config, optimizer)
    ts_best_top1 = 0.0
    ss_best_top1 = 0.0
    is_master = get_rank() == 0
    checkpointer = Checkpointer(student_model, optimizer, scheduler, student_config.output_dir, is_master)
    """ fixed student vision encoder and only ditillate student text model """
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        student_model.train()

        for n_iter, batch in enumerate(train_loader):
            
            batch = {k: v.to(device) for k, v in batch.items()}

            # 固定学生图像编码器
            i_feat, t_feat = student_model(batch, vision_grad=False, text_grad=True)
            # i_feat = student_model.encode_image(batch['images']) 
            # with torch.no_grad(): 
            #     # fix student text encoder   
            #     t_feat = student_model.encode_text(batch['caption_ids'])          

            rep_visual_loss = 0.
            rep_text_loss = 0.
            att_visual_loss = 0.
            att_text_loss = 0.        
            value_visual_loss = 0.
            value_text_loss = 0.
            
            distillation_pred_loss = 0.    
            task_loss = 0.
            
            if student_config.distillation:
                # Gather student states extracted by hooks
                # temp_model = unwrap_ddp(student_model)
                student_reps = flatten_states(student_model.distill_states_dict, "hidden_states")
                student_atts = flatten_states(student_model.distill_states_dict, "attention_scores")        
                student_values = flatten_states(student_model.distill_states_dict, "value_states")
                student_pred_states = flatten_states(student_model.distill_states_dict, "pred_states")
            
                #No gradient for teacher training
                with torch.no_grad():
                    teacher_model(batch)
                # Gather teacher states extracted by hooks
                # temp_model = unwrap_ddp(teacher_model)
                teacher_reps = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "hidden_states")]
                teacher_atts = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "attention_scores")]        
                teacher_values = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "value_states")]
                teacher_pred_states = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "pred_states")]
 
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
                        new_teacher_visual_pred = [teacher_pred_states[-1]]
                        new_student_visual_pred = [student_pred_states[-1]]
                        new_teacher_text_pred = [teacher_pred_states[0]]                
                        new_student_text_pred = [student_pred_states[0]]  
                                    
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
                        new_teacher_visual_pred = [teacher_pred_states[-1]]
                        new_student_visual_pred = [student_pred_states[-1]]
                        new_teacher_text_pred = [teacher_pred_states[0]]                
                        new_student_text_pred = [student_pred_states[0]]
                
                # 模型最后输出层的知识蒸馏
                if student_config.distillation_config["use_pred_states"]:
                    # 只蒸馏文本编码器, MoTIS 算法
                    if "MoTIS" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_MoTIS_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config, vision_distill=False, text_distill=True)
                    if "inter_ts_kl_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_tch_stu_kl_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config, vision_distill=False, text_distill=True)
                    # inter-model stu-stu learning: 固定学生图像编码器，蒸馏学生文本编码器
                    if "inter_ss_kl_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_stu_stu_kl_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config)
                    # ConaCLIP 算法
                    if "intra_ss_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_intra_model_stu_stu_sd_loss(new_student_visual_pred[0], new_student_text_pred[0],
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        vision_distill=False, text_distill=True)
                    if "inter_ss_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_model_stu_stu_sd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0])

                    if "intra_ts_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_intra_model_tch_stu_sd_loss(new_student_visual_pred[0], new_student_text_pred[0],
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0], 
                                                                        vision_distill=False, text_distill=True)
                    # sym-sd
                    if "intra_ts_sym_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_intra_model_tch_stu_sym_sd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0])
                    # sym-kl-div
                    if "inter_ts_sym_kl_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_tch_stu_sym_kl_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0], student_config) 

                # 加入 TBPS 任务损失计算
                if student_config.distillation_config["use_task_loss"]: 
                    # 共享教师模型的图像编码器 
                    if "ts_itc" in student_config.distillation_config["task_loss"]:
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        task_loss += objectives.compute_itc(new_teacher_visual_pred[0], new_student_text_pred[0], logit_scale)
                    if "ts_sdm" in student_config.distillation_config["task_loss"]:                                         
                        # 共享教师模型的图像编码器，进计算文本编码器的 sdm loss
                        task_loss += compute_inter_tch_stu_sdm_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                new_teacher_visual_pred[0],  new_teacher_text_pred[0], 
                                                                student_config, batch['pids'], vision_loss=False, text_loss=True)  
                    # 学生图像和文本编码器联合计算损失
                    if "ss_itc" in student_config.distillation_config["task_loss"]:
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        task_loss += objectives.compute_itc(new_student_visual_pred[0], new_student_text_pred[0], logit_scale) 
                    if "ss_sdm" in student_config.distillation_config["task_loss"]: 
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        task_loss = objectives.compute_sdm(new_student_visual_pred[0], new_student_text_pred[0], batch['pids'], logit_scale)                                
            else:
                # sdm loss
                logit_scale = torch.ones([]) * (1 / student_config.temperature)
                task_loss = objectives.compute_sdm(i_feat, t_feat, batch['pids'], logit_scale)

            if student_config.distillation:
                if student_config.distillation_cosin_weight:
                    # loss weight 计算
                    alpha = tch_stu_loss_weight(epoch, init_alpha=0.9, last_alpha=0.5, start_epoch=20, end_epoch=50)
                elif student_config.distillation_constant_weight:
                    alpha = student_config.distillation_weight
                else:                    
                    # loss weight 计算            
                    if epoch < 10:
                        alpha = 0.9
                    elif start_epoch < 20:
                        alpha = 0.6
                    elif start_epoch < 40:
                        alpha = 0.3
                    else:
                        alpha = 0.1
            else:
                alpha = 0.0

            # loss 计算
            total_loss = att_visual_loss + att_text_loss + \
                        rep_visual_loss + rep_text_loss + \
                        value_visual_loss + value_text_loss + \
                        alpha * distillation_pred_loss + (1 - alpha) * task_loss
        
            batch_size = batch['images'].shape[0]
            meters['loss'].update(get_loss(total_loss), batch_size)
            meters['task_loss'].update(get_loss(task_loss), batch_size)
            meters['distillation_pred_loss'].update(get_loss(distillation_pred_loss), batch_size)
            # meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            # meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            # meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            # meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            # meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            # meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.6f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        # 绘制到 TensorBoard 进行可视化
        # tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        # tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        # for k, v in meters.items():
        #     if v.avg > 0:
        #         tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Share teacher vision encoder for evaluation ...")
                logger.info("Validation Results - Epoch: {}".format(epoch))
                top1 = evaluator.eval(student_model.eval(), teacher_model.eval(), use_tch_text=False, use_tch_image=True)
                if ts_best_top1 < top1:
                    ts_best_top1 = top1
                    arguments["ts_best_top1_epoch"] = epoch

                logger.info("Distillation student model evaluation ...")
                logger.info("Validation Results - Epoch: {}".format(epoch))
                top1 = evaluator.eval(student_model.eval())

                torch.cuda.empty_cache()
                if ss_best_top1 < top1:
                    ss_best_top1 = top1
                    arguments["ss_best_top1_epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"ts_best_top1 R1: {ts_best_top1} at epoch {arguments['ts_best_top1_epoch']}")
        logger.info(f"ss_best_top1 R1: {ss_best_top1} at epoch {arguments['ss_best_top1_epoch']}")







def do_train_inline_distillation(start_epoch, student_model, student_config, 
                        train_loader, evaluator, optimizer, scheduler, 
                        checkpointer, logger,
                        teacher_model=None, teacher_config=None, project=None):

    log_period = int(student_config.log_period)
    eval_period = int(student_config.eval_period) 
    # device = "cuda"
    device = next(student_model.parameters()).device
    num_epoch = student_config.num_epoch
    
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    # logger = logging.getLogger("Distillation.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "task_loss": AverageMeter(),
        "distillation_pred_loss": AverageMeter(),
        # "itc_loss": AverageMeter(),
        # "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=student_config.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        student_model.train()

        for n_iter, batch in enumerate(train_loader):
            
            batch = {k: v.to(device) for k, v in batch.items()}

            # batch_size = batch['images'].shape[0]
            # pid = batch['pids']
            # pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
            # pid_dist = pid - pid.t()
            # labels = (pid_dist == 0).float()
            # print(labels)

            i_feat, t_feat = student_model(batch)

            rep_visual_loss = 0.
            rep_text_loss = 0.
            att_visual_loss = 0.
            att_text_loss = 0.        
            value_visual_loss = 0.
            value_text_loss = 0.
            
            distillation_pred_loss = 0.    
            task_loss = 0.
            
            if student_config.distillation:
                # Gather student states extracted by hooks
                # temp_model = unwrap_ddp(student_model)
                student_reps = flatten_states(student_model.distill_states_dict, "hidden_states")
                student_atts = flatten_states(student_model.distill_states_dict, "attention_scores")        
                student_values = flatten_states(student_model.distill_states_dict, "value_states")
                student_pred_states = flatten_states(student_model.distill_states_dict, "pred_states")
            
                #No gradient for teacher training
                with torch.no_grad():
                    teacher_model(batch)
                # Gather teacher states extracted by hooks
                # temp_model = unwrap_ddp(teacher_model)
                teacher_reps = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "hidden_states")]
                teacher_atts = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "attention_scores")]        
                teacher_values = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "value_states")]
                teacher_pred_states = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "pred_states")]
 

                

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
                        new_teacher_visual_pred = [teacher_pred_states[-1]]
                        new_student_visual_pred = [student_pred_states[-1]]
                        new_teacher_text_pred = [teacher_pred_states[0]]                
                        new_student_text_pred = [student_pred_states[0]]  
                                    
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
                        new_teacher_visual_pred = [teacher_pred_states[-1]]
                        new_student_visual_pred = [student_pred_states[-1]]
                        new_teacher_text_pred = [teacher_pred_states[0]]                
                        new_student_text_pred = [student_pred_states[0]]

                # if student_config.distillation_config["use_hidden_states"]:
                #     # if use_projection:
                #     #     rep_loss = transformer_losses.compute_loss(project(new_student_reps), new_teacher_reps, loss_name="hidden_state_loss")
                #     # else:
                #     #     rep_loss = transformer_losses.compute_loss(new_student_reps, new_teacher_reps, loss_name="hidden_state_loss")
                #     if use_projection_visual:
                #         rep_visual_loss = visual_transformer_losses.compute_loss(project_visual_hidden_state(new_student_visual_reps), new_teacher_visual_reps, loss_name="hidden_state_loss")
                #     else:
                #         rep_visual_loss = visual_transformer_losses.compute_loss(new_student_visual_reps, new_teacher_visual_reps, loss_name="hidden_state_loss")

                #     if use_projection_text:
                #         rep_text_loss = text_transformer_losses.compute_loss(project_text_hidden_state(new_student_text_reps), new_teacher_text_reps, loss_name="hidden_state_loss")
                #     else:
                #         rep_text_loss = text_transformer_losses.compute_loss(new_student_text_reps, new_teacher_text_reps, loss_name="hidden_state_loss")

                # if student_config.distillation_config["use_attention_scores"]:
                #     # att_loss = transformer_losses.compute_loss(student_atts, new_teacher_atts, loss_name="attention_loss")
                #     att_visual_loss = visual_transformer_losses.compute_loss(student_visual_atts, new_teacher_visual_atts, loss_name="attention_loss")
                #     att_text_loss = text_transformer_losses.compute_loss(student_text_atts, new_teacher_text_atts, loss_name="attention_loss")
                        
                # if student_config.distillation_config["use_value_states"]:
                #     # value_loss = transformer_losses.compute_loss(student_values, new_teacher_values, loss_name="value_state_loss")
                    
                #     # value_visual_loss = visual_transformer_losses.compute_loss(student_visual_values, new_teacher_visual_values, loss_name="value_state_loss")
                #     # value_text_loss = text_transformer_losses.compute_loss(student_text_values, new_teacher_text_values, loss_name="value_state_loss")
                #     if use_projection_visual:
                #         value_visual_loss = visual_transformer_losses.compute_loss(project_visual_value_state(student_visual_values), new_teacher_visual_values, loss_name="value_state_loss")                
                #     else:
                #         value_visual_loss = visual_transformer_losses.compute_loss(student_visual_values, new_teacher_visual_values, loss_name="value_state_loss")

                #     if use_projection_text:
                #         value_text_loss = text_transformer_losses.compute_loss(project_text_value_state(student_text_values), new_teacher_text_values, loss_name="value_state_loss")
                #     else:
                #         value_text_loss = text_transformer_losses.compute_loss(student_text_values, new_teacher_text_values, loss_name="value_state_loss")

                # 模型最后输出层的知识蒸馏
                if student_config.distillation_config["use_pred_states"]:
                    if "ssdm" in student_config.distillation_config["pred_states_loss"]: 
                        distillation_pred_loss += distillation_soft_sdm_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config, batch['pids'])
                    if "MoTIS" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_MoTIS_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                        student_config)
                                        
                    
                    if "fd" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_fd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0])
                    

                    if "intra_ss_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_intra_model_stu_stu_sd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0])
                    if "inter_ss_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_model_stu_stu_sd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0])
                    if "intra_ts_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_intra_model_tch_stu_sd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0])
                    if "inter_ts_sd_loss" in student_config.distillation_config["pred_states_loss"]:
                        distillation_pred_loss += distillation_inter_model_tch_stu_sd_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                        new_teacher_visual_pred[0],  new_teacher_text_pred[0])

                    if "reid_MoTIS_loss" in student_config.distillation_config["pred_states_loss"]:
                        vision_distill = student_config.reid_MoTIS_vision
                        text_distill = student_config.reid_MoTIS_text
                        distillation_pred_loss += distillation_reid_MoTIS_loss(new_student_visual_pred[0], new_student_text_pred[0], 
                                                                            new_teacher_visual_pred[0],  new_teacher_text_pred[0],
                                                                            student_config, batch['pids'],
                                                                            vision_distill=vision_distill, text_distill=text_distill)

                # 加入 TBPS 任务 sdm loss 损失计算
                if student_config.distillation_config["use_task_loss"]:  
                    if student_config.distillation_config["task_loss"] == "sdm":
                        logit_scale = torch.ones([]) * (1 / student_config.temperature)
                        task_loss = objectives.compute_sdm(new_student_visual_pred[0], new_student_text_pred[0], batch['pids'], logit_scale)                
            else:
                if "sdm" in student_config.loss_names:
                    # sdm loss
                    logit_scale = torch.ones([]) * (1 / student_config.temperature)
                    task_loss += objectives.compute_sdm(i_feat, t_feat, batch['pids'], logit_scale)
                if "smse" in student_config.loss_names:
                    # sd loss
                    task_loss += objectives.compute_self_mse(i_feat, t_feat, batch['pids'], image_mse=True, text_mse=True, altha=0.8)


            # loss 计算
            total_loss = att_visual_loss + att_text_loss + \
                        rep_visual_loss + rep_text_loss + \
                        value_visual_loss + value_text_loss + \
                        distillation_pred_loss + \
                        task_loss
        
            batch_size = batch['images'].shape[0]
            meters['loss'].update(get_loss(total_loss), batch_size)
            meters['task_loss'].update(get_loss(task_loss), batch_size)
            meters['distillation_pred_loss'].update(get_loss(distillation_pred_loss), batch_size)
            # meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            # meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            # meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            # meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            # meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            # meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.6f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        # 绘制到 TensorBoard 进行可视化
        # tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        # tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        # for k, v in meters.items():
        #     if v.avg > 0:
        #         tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if student_config.distributed:
                    top1 = evaluator.eval(student_model.module.eval())
                else:
                    top1 = evaluator.eval(student_model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")

#========================================================================================================
def do_train_distillation_kd(start_epoch, student_model, student_config, 
                        train_loader, evaluator, optimizer, scheduler, 
                        checkpointer, logger,
                        teacher_model=None, teacher_config=None, project=None):

    log_period = int(student_config.log_period)
    eval_period = int(student_config.eval_period) 
    device = next(student_model.parameters()).device
    num_epoch = student_config.num_epoch
    
    if teacher_model is None:
        raise ValueError("CLIP-KD distillation requires teacher_model != None")

    # Freeze teacher model parameters
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    # -------------------------
    # Build CLIP-KD loss
    # -------------------------
    clipkd_loss_fn = CLIPKDLoss(
        temperature=getattr(student_config, "clipkd_temperature", 0.07),
        lambda_fd=getattr(student_config, "clipkd_fd_weight", 2000.0),
        lambda_icl=getattr(student_config, "clipkd_icl_weight", 1.0),
        lambda_crd=getattr(student_config, "clipkd_crd_weight", 1.0),
    ).to(device)

    use_fd = getattr(student_config, "use_fd", True)
    use_icl = getattr(student_config, "use_icl", True)
    use_crd = getattr(student_config, "use_crd", True)

    clipkd_weight = float(getattr(student_config, "clipkd_weight", 1.0))
    task_weight = float(getattr(student_config, "lambda_task", 1.0))

    logger.info("=" * 60)
    logger.info("CLIP-KD Training Configuration:")
    logger.info(f"  use_fd={use_fd}, use_icl={use_icl}, use_crd={use_crd}")
    logger.info(f"  clipkd_temperature={getattr(student_config, 'clipkd_temperature', 0.07)}")
    logger.info(f"  clipkd_fd_weight(lambda_fd)={getattr(student_config, 'clipkd_fd_weight', 2000.0)}")
    logger.info(f"  clipkd_icl_weight(lambda_icl)={getattr(student_config, 'clipkd_icl_weight', 1.0)}")
    logger.info(f"  clipkd_crd_weight(lambda_crd)={getattr(student_config, 'clipkd_crd_weight', 1.0)}")
    logger.info(f"  clipkd_weight(total)={clipkd_weight}")
    logger.info(f"  task_weight={task_weight}")
    logger.info("=" * 60)

    # Initialize meters for tracking losses
    meters = {
        "loss": AverageMeter(),
        "task_loss": AverageMeter(),
        "clipkd_loss": AverageMeter(),
        "fd_loss": AverageMeter(),
        "icl_loss": AverageMeter(),
        "crd_loss": AverageMeter(),
    }

    tb_writer = None
    if get_rank() == 0:
        tb_writer = SummaryWriter(log_dir=student_config.output_dir)

    arguments = {"num_epoch": num_epoch, "iteration": 0, "epoch": start_epoch}
    best_top1 = 0.0

    # Helper function
    def _to_device(x):
        return x.to(device, non_blocking=True) if torch.is_tensor(x) else x

    def _get_lr():
        if hasattr(scheduler, "get_lr"):
            lr_list = scheduler.get_lr()
            return float(lr_list[0]) if isinstance(lr_list, (list, tuple)) else float(lr_list)
        if hasattr(scheduler, "get_last_lr"):
            lr_list = scheduler.get_last_lr()
            return float(lr_list[0]) if isinstance(lr_list, (list, tuple)) else float(lr_list)
        return float(optimizer.param_groups[0]["lr"])

    logger.info("Start training (CLIP-KD)")

    # Training loop
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for m in meters.values():
            m.reset()

        student_model.train()
        teacher_model.eval()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: _to_device(v) for k, v in batch.items()}

            # ✅ 关键修改：使用与 do_train_distillation 相同的方式获取特征
            # Forward student to populate distill_states_dict
            _ = student_model(batch)
            
            # Extract student features from pred_states (same as do_train_distillation)
            student_pred_states = flatten_states(student_model.distill_states_dict, "pred_states")
            
            # Forward teacher (no grad)
            with torch.no_grad():
                _ = teacher_model(batch)
            
            # Extract teacher features from pred_states
            teacher_pred_states = [i.detach() for i in flatten_states(teacher_model.distill_states_dict, "pred_states")]

            # ✅ 按照 do_train_distillation 的方式提取图像和文本特征
            # pred_states 的结构：通常是 [text_feat, ..., image_feat] 或类似顺序
            # 根据你的原始代码，pred_states[-1] 是 image，pred_states[0] 是 text
            s_i_feat = student_pred_states[-1]  # 学生图像特征
            s_t_feat = student_pred_states[0]   # 学生文本特征
            t_i_feat = teacher_pred_states[-1]  # 教师图像特征
            t_t_feat = teacher_pred_states[0]   # 教师文本特征

            # Handle embedding dimension mismatch if needed
            if project is not None and callable(project):
                if s_i_feat.shape[-1] != t_i_feat.shape[-1]:
                    s_i_feat = project(s_i_feat)
                if s_t_feat.shape[-1] != t_t_feat.shape[-1]:
                    s_t_feat = project(s_t_feat)

            # Task loss
            task_loss = 0.0
            if task_weight != 0.0:
                if "itc" in student_config.loss_names:
                    logit_scale = torch.ones([], device=device) * (1.0 / float(getattr(student_config, "temperature", 1.0)))
                    task_loss += objectives.compute_itc(s_i_feat, s_t_feat, logit_scale)
                if "sdm" in student_config.loss_names:
                    logit_scale = torch.ones([], device=device) * (1.0 / float(getattr(student_config, "temperature", 1.0)))
                    task_loss += objectives.compute_sdm(s_i_feat, s_t_feat, batch["pids"], logit_scale)
                if "smse" in student_config.loss_names:
                    task_loss += objectives.compute_self_mse(s_i_feat, s_t_feat, batch["pids"], image_mse=True, text_mse=True, altha=0.8)

            # CLIP-KD distillation loss
            clipkd_kd_loss, clipkd_dict = clipkd_loss_fn(
                student_image_features=s_i_feat,
                student_text_features=s_t_feat,
                teacher_image_features=t_i_feat,
                teacher_text_features=t_t_feat,
                use_fd=use_fd,
                use_icl=use_icl,
                use_crd=use_crd,
            )

            # Total loss
            total_loss = task_weight * task_loss + clipkd_weight * clipkd_kd_loss

            # Update meters
            batch_size = batch["images"].shape[0] if "images" in batch else s_i_feat.shape[0]
            meters["loss"].update(get_loss(total_loss), batch_size)
            meters["task_loss"].update(get_loss(task_loss) if isinstance(task_loss, torch.Tensor) else float(task_loss), batch_size)
            meters["clipkd_loss"].update(get_loss(clipkd_kd_loss), batch_size)

            # Update individual losses
            if isinstance(clipkd_dict, dict):
                if "fd_loss" in clipkd_dict:
                    meters["fd_loss"].update(float(clipkd_dict["fd_loss"]), batch_size)
                if "icl_loss" in clipkd_dict:
                    meters["icl_loss"].update(float(clipkd_dict["icl_loss"]), batch_size)
                if "crd_loss" in clipkd_dict:
                    meters["crd_loss"].update(float(clipkd_dict["crd_loss"]), batch_size)

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            arguments["iteration"] += 1

            # Logging
            if (n_iter + 1) % log_period == 0:
                lr = _get_lr()
                info = f"Epoch[{epoch}] Iter[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.avg > 0:
                        info += f", {k}: {v.avg:.6f}"
                info += f", Lr: {lr:.2e}"
                logger.info(info)

        # End of epoch
        scheduler.step()

        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / max(1, (n_iter + 1))
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

            # TensorBoard logging
            if tb_writer is not None:
                tb_writer.add_scalar("lr", _get_lr(), epoch)
                for k, v in meters.items():
                    if v.avg > 0:
                        tb_writer.add_scalar(k, v.avg, epoch)

        # Evaluation and save best model
        if (epoch % eval_period == 0) and (get_rank() == 0):
            logger.info("=" * 80)
            logger.info(f"Validation Results - Epoch: {epoch}")

            eval_model = student_model.module if getattr(student_config, "distributed", False) else student_model
            
            # ✅ 评估并生成表格
            results = evaluator.eval(eval_model.eval())
            
            # 如果 evaluator.eval 返回的是字典，生成表格
            if isinstance(results, dict):
                table = print_evaluation_table(results, task_name="t2i")
                logger.info("\n" + str(table))
                top1 = results.get('R1', 0.0)
            else:
                # 如果只返回 top1 值
                top1 = results

            # Cache clean
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            if top1 > best_top1:
                best_top1 = top1
                arguments["epoch"] = epoch
                checkpointer.save("best", **arguments)
            
            logger.info("=" * 80)

    if tb_writer is not None:
        tb_writer.close()

    if get_rank() == 0:
        logger.info(f"Best R1: {best_top1} at epoch {arguments.get('epoch', -1)}")


def print_evaluation_table(results, task_name="t2i"):
    """
    使用 PrettyTable 打印评估结果
    
    Args:
        results: 包含评估指标的字典，例如 {'R1': 56.498, 'R5': 78.021, ...}
        task_name: 任务名称，默认 "t2i"
    """
    table = PrettyTable()
    table.field_names = ["task", "R1", "R5", "R10", "mAP", "mINP"]
    
    # 添加数据行
    table.add_row([
        task_name,
        f"{results.get('R1', 0.0):.3f}",
        f"{results.get('R5', 0.0):.3f}",
        f"{results.get('R10', 0.0):.3f}",
        f"{results.get('mAP', 0.0):.3f}",
        f"{results.get('mINP', 0.0):.3f}"
    ])
    
    return table
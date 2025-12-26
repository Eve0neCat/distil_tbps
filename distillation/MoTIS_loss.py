import torch
import torch.nn.functional as F


def distillation_MoTIS_loss(student_image_fetures, student_text_fetures, 
                            teacher_image_fetures, teacher_text_fetures, 
                            student_config, vision_distill=True, text_distill=True):
    """
    MoTIS loss:
        calculate text features itc loss between student and teacher
        calculate image features itc loss between student and teacher
    """
    batch_size = student_image_fetures.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(student_image_fetures.device)
    
    # temperature for distillation
    logit_scale = 1. / student_config.distillation_MoTIS_temperature

    if vision_distill:
        # normalized teacher and student vision features 
        teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
        student_image_norm = student_image_fetures / student_image_fetures.norm(dim=-1, keepdim=True)
        # vision distillation
        logits_image_sim = logit_scale * student_image_norm @ teacher_image_norm.t()
        loss_i = F.cross_entropy(logits_image_sim, labels)

    if text_distill:
        # normalized teacher and student vision features 
        teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)
        student_text_norm = student_text_fetures / student_text_fetures.norm(dim=-1, keepdim=True)
        # text distillation
        logits_text_sim = logit_scale * student_text_norm @ teacher_text_norm.t()
        loss_t = F.cross_entropy(logits_text_sim, labels)

    if vision_distill and not text_distill:
        return loss_i
    elif not vision_distill and text_distill:
        return loss_t
    elif vision_distill and text_distill:
        return (loss_t + loss_i) / 2.
    else:
        raise ValueError("No distillation task is selected")

def distillation_reid_MoTIS_loss(student_image_fetures, student_text_fetures, 
                                teacher_image_fetures, teacher_text_fetures, 
                                student_config, pid, vision_distill=True, text_distill=True):
    """
    MoTIS loss:
        calculate text features itc loss between student and teacher
        calculate image features itc loss between student and teacher
    """
    batch_size = student_image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()
    
    # temperature for distillation
    logit_scale = 1. / student_config.distillation_reid_MoTIS_temperature

    if vision_distill:
        # normalized teacher and student vision features 
        teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
        student_image_norm = student_image_fetures / student_image_fetures.norm(dim=-1, keepdim=True)
        # vision distillation
        logits_image_sim = logit_scale * student_image_norm @ teacher_image_norm.t()
        loss_i = F.cross_entropy(logits_image_sim, labels)

    if text_distill:
        # normalized teacher and student vision features 
        teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)
        student_text_norm = student_text_fetures / student_text_fetures.norm(dim=-1, keepdim=True)
        # text distillation
        logits_text_sim = logit_scale * student_text_norm @ teacher_text_norm.t()
        loss_t = F.cross_entropy(logits_text_sim, labels)

    if vision_distill and not text_distill:
        return loss_i
    elif not vision_distill and text_distill:
        return loss_t
    elif vision_distill and text_distill:
        return (loss_t + loss_i) / 2.
    else:
        raise ValueError("No distillation task is selected")



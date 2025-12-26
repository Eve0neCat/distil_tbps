import torch
import torch.nn.functional as F


def distillation_cross_kd_loss(student_image_fetures, student_text_fetures,
                        teacher_image_fetures, teacher_text_fetures,
                        student_config):
    """
    软标签交叉熵损失: cross_kd_loss
    """
    T = student_config.distillation_cross_kd_temperature
    logit_scale = 1. / T
    
    teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
    teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)

    student_image_norm = student_image_fetures / student_image_fetures.norm(dim=1, keepdim=True)
    student_text_norm = student_text_fetures / student_text_fetures.norm(dim=1, keepdim=True)

    t2i_soft_label = logit_scale * teacher_text_norm @ teacher_image_norm.t()
    i2t_soft_label = t2i_soft_label.t()

    t2i_sim = logit_scale * student_text_norm @ student_image_norm.t()
    i2t_sim = t2i_sim.t()

    t2i_teacher_probs = F.softmax(t2i_soft_label, dim=-1)
    i2t_teacher_probs = F.softmax(i2t_soft_label, dim=-1)

    # 软标签交叉熵损失
    loss = (F.cross_entropy(t2i_sim, t2i_teacher_probs) + F.cross_entropy(i2t_sim, i2t_teacher_probs)) * (T**2) / 2.

    return loss


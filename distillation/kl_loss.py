import torch
import torch.nn.functional as F


def compute_sdm(image_fetures, text_fetures, pid, temperature, epsilon=1e-8):
    """
    Similarity Distribution Matching, is reversal KL divergence
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    logit_scale = 1. / temperature
    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def distillation_cross_kl_loss(student_image_fetures, student_text_fetures,
                        teacher_image_fetures, teacher_text_fetures,
                        student_config):
    """
    cross-model learning: KL divergence
    Hinton KD: 使用正向 KL 散度，计算两个模型之间的相似性
    简称: cross_kl_loss
    """
    T = student_config.distillation_cross_kl_temperature
    logit_scale = 1. / T
    
    teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
    teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)

    student_image_norm = student_image_fetures / student_image_fetures.norm(dim=1, keepdim=True)
    student_text_norm = student_text_fetures / student_text_fetures.norm(dim=1, keepdim=True)

    t2i_soft_label = logit_scale * teacher_text_norm @ teacher_image_norm.t()
    i2t_soft_label = t2i_soft_label.t()

    t2i_sim = logit_scale * student_text_norm @ student_image_norm.t()
    i2t_sim = t2i_sim.t()

    # 正向 KL 散度
    loss = (F.kl_div(F.log_softmax(t2i_sim, dim=1), F.softmax(t2i_soft_label, dim=1), reduction='batchmean') + 
            F.kl_div(F.log_softmax(i2t_sim, dim=1), F.softmax(i2t_soft_label, dim=1), reduction='batchmean')) * (T**2) / 2.

    return loss

def distillation_cross_rkl_loss(student_image_fetures, student_text_fetures,
                        teacher_image_fetures, teacher_text_fetures,
                        student_config):
    """
    cross-model learning: RKL divergence
    Hinton KD: 使用反向 KL 散度，计算两个模型之间的相似性
    简称: cross_rkl_loss
    """
    T = student_config.distillation_cross_rkl_temperature
    logit_scale = 1. / T
    
    teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
    teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)

    student_image_norm = student_image_fetures / student_image_fetures.norm(dim=1, keepdim=True)
    student_text_norm = student_text_fetures / student_text_fetures.norm(dim=1, keepdim=True)

    t2i_soft_label = logit_scale * teacher_text_norm @ teacher_image_norm.t()
    i2t_soft_label = t2i_soft_label.t()

    t2i_sim = logit_scale * student_text_norm @ student_image_norm.t()
    i2t_sim = t2i_sim.t()

    # 正向 KL 散度
    loss = (F.kl_div(F.log_softmax(t2i_soft_label, dim=1), F.softmax(t2i_sim, dim=1), reduction='batchmean') + 
            F.kl_div(F.log_softmax(i2t_soft_label, dim=1), F.softmax(i2t_sim, dim=1), reduction='batchmean')) * (T**2) / 2.

    return loss

def distillation_inter_stu_stu_kl_loss(student_image_fetures, student_text_fetures, 
                                    teacher_image_fetures, teacher_text_fetures, 
                                    student_config):
    """
    inter-model stu-stu learning: KL divergence
    简称: inter_ss_kl_loss
    """    
    teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
    teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)

    student_image_norm = student_image_fetures / student_image_fetures.norm(dim=1, keepdim=True)
    student_text_norm = student_text_fetures / student_text_fetures.norm(dim=1, keepdim=True)

    logit_scale = 1. / student_config.distillation_inter_ss_kl_temperature
    
    # KL 1
    t2i_sim = logit_scale * student_text_norm @ student_image_norm.t()
    label_t2i_sim = logit_scale * teacher_text_norm @ teacher_image_norm.t()

    t2i_pred = F.softmax(t2i_sim, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(t2i_sim, dim=1) - F.log_softmax(label_t2i_sim))

    # KL 2
    i2t_sim = logit_scale * student_image_norm @ student_text_norm.t()
    label_i2t_sim = logit_scale * teacher_image_norm @ teacher_text_norm.t()
    
    i2t_pred = F.softmax(i2t_sim, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(i2t_sim, dim=1) - F.log_softmax(label_i2t_sim, dim=1))

    loss = torch.mean(torch.sum(t2i_loss, dim=1)) + torch.mean(torch.sum(i2t_loss, dim=1)) 

    return loss



def distillation_inter_tch_stu_kl_loss(student_image_fetures, student_text_fetures, 
                                    teacher_image_fetures, teacher_text_fetures, 
                                    student_config, vision_distill=True, text_distill=True):
    """
    inter-model tch-stu learning: KL divergence
    简称: inter_ts_kl_loss
    """    
    teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
    teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)

    student_image_norm = student_image_fetures / student_image_fetures.norm(dim=1, keepdim=True)
    student_text_norm = student_text_fetures / student_text_fetures.norm(dim=1, keepdim=True)

    logit_scale = 1. / student_config.distillation_inter_ts_kl_temperature

    # KL 1
    if text_distill:
        t2i_sim = logit_scale * student_text_norm @ teacher_image_norm.t()
        label_t2i_sim = logit_scale * teacher_text_norm @ teacher_image_norm.t()

        t2i_pred = F.softmax(t2i_sim, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(t2i_sim, dim=1) - F.log_softmax(label_t2i_sim))
        
        loss_t = torch.mean(torch.sum(t2i_loss, dim=1))

    # KL 2
    if vision_distill:
        i2t_sim = logit_scale * student_image_norm @ teacher_text_norm.t()
        label_i2t_sim = logit_scale * student_image_norm @ teacher_text_norm.t()
        
        i2t_pred = F.softmax(i2t_sim, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(i2t_sim, dim=1) - F.log_softmax(label_i2t_sim, dim=1))

        loss_i = torch.mean(torch.sum(i2t_loss, dim=1)) 

    if vision_distill and not text_distill:
        return loss_i
    elif not vision_distill and text_distill:
        return loss_t
    elif vision_distill and text_distill:
        return (loss_t + loss_i) / 2.
    else:
        raise ValueError("No distillation task is selected")

def distillation_inter_tch_stu_sym_kl_loss(student_image_fetures, student_text_fetures, 
                                    teacher_image_fetures, teacher_text_fetures, 
                                    student_config):
    """
    inter-model tch-stu learning: Symmetric KL divergence
    简称: inter_ts_sym_kl_loss
    """    
    teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
    teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)

    student_image_norm = student_image_fetures / student_image_fetures.norm(dim=1, keepdim=True)
    student_text_norm = student_text_fetures / student_text_fetures.norm(dim=1, keepdim=True)

    logit_scale = 1. / student_config.distillation_inter_ts_sym_kl_temperature

    # KL 1
    t2i_cosine_theta = logit_scale * student_text_norm @ teacher_image_norm.t()
    i2t_cosine_theta = logit_scale * student_image_norm @ teacher_text_norm.t()

    t2i_pred = F.softmax(t2i_cosine_theta, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(t2i_cosine_theta, dim=1) - F.log_softmax(i2t_cosine_theta))

    # KL 2
    stu2tch_i2t_cosine_theta = logit_scale * student_image_norm @ teacher_text_norm.t()
    stu2tch_t2i_cosine_theta = logit_scale * student_text_norm @ teacher_image_norm.t()
    
    i2t_pred = F.softmax(stu2tch_i2t_cosine_theta, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(stu2tch_i2t_cosine_theta, dim=1) - F.log_softmax(stu2tch_t2i_cosine_theta, dim=1))

    loss = torch.mean(torch.sum(t2i_loss, dim=1)) + torch.mean(torch.sum(i2t_loss, dim=1)) 

    return loss

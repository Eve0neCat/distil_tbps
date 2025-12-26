import torch
import torch.nn.functional as F


def compute_sdm(image_fetures, text_fetures, pid, temperature, epsilon=1e-8):
    """
    Similarity Distribution Matching
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

def compute_inter_tch_stu_sdm_loss(student_image_fetures, student_text_fetures, 
                            teacher_image_fetures, teacher_text_fetures, 
                            student_config, pid, vision_loss=True, text_loss=True, epsilon=1e-8):
    """
    简称: inter_ts_sdm_loss with hard labels
    Similarity Distribution Matching between teacher and student
    """
    batch_size = student_image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    logit_scale = 1. / student_config.distillation_inter_tch_stu_sdm_temperature

    if vision_loss:
        teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)
        student_image_norm = student_image_fetures / student_image_fetures.norm(dim=1, keepdim=True)
        i2t_cosine_theta = logit_scale * student_image_norm @ teacher_text_norm.t()
        i2t_pred = F.softmax(i2t_cosine_theta, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(i2t_cosine_theta, dim=1) - torch.log(labels_distribute + epsilon))
        loss_i = torch.mean(torch.sum(i2t_loss, dim=1))

    if text_loss:
        teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)        
        student_text_norm = student_text_fetures / student_text_fetures.norm(dim=1, keepdim=True)
        t2i_cosine_theta = logit_scale * student_text_norm @ teacher_image_norm.t()
        t2i_pred = F.softmax(t2i_cosine_theta, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(t2i_cosine_theta, dim=1) - torch.log(labels_distribute + epsilon))       
        loss_t = torch.mean(torch.sum(t2i_loss, dim=1))

    if vision_loss and not text_loss:
        return loss_i
    elif not vision_loss and text_loss:
        return loss_t
    elif vision_loss and text_loss:
        return (loss_t + loss_i) / 2.
    else:
        raise ValueError("No distillation task is selected")







def distillation_soft_sdm_loss(student_image_fetures, student_text_fetures, 
                            teacher_image_fetures, teacher_text_fetures, 
                            student_config, pid, epsilon=1e-8):
    """
    Similarity Distribution Matching mixed soft label from teacher and hard label
    混合 hard 标签和 soft 标签
    """
    batch_size = student_image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    # teacher normalized features for soft labels
    teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
    teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)

    # soft label from teacher
    t2i_sim = teacher_text_norm @ teacher_image_norm.t()    
    i2t_sim = t2i_sim.t()

    # normalize the mixed soft label distribution
    labels_i2t = F.softmax(t2i_sim, dim=1)
    labels_t2i = F.softmax(i2t_sim, dim=1)

    # construct mixed soft label
    alptha = student_config.soft_sdm_loss_alpha
    labels_t2i = labels * alptha + t2i_sim * (1 - alptha)
    labels_i2t = labels * alptha + i2t_sim * (1 - alptha)

    # calculate cosine similarity for student
    student_image_norm = student_image_fetures / student_image_fetures.norm(dim=1, keepdim=True)
    student_text_norm = student_text_fetures / student_text_fetures.norm(dim=1, keepdim=True)
    logit_scale = 1. / student_config.distillation_ssdm_temperature
    t2i_cosine_theta = logit_scale * student_text_norm @ student_image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    # calculate kl loss
    i2t_pred = F.softmax(i2t_cosine_theta, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(i2t_cosine_theta, dim=1) - torch.log(labels_i2t + epsilon))
    
    t2i_pred = F.softmax(t2i_cosine_theta, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(t2i_cosine_theta, dim=1) - torch.log(labels_t2i + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def distillation_intra_tch_stu_sdm_loss(student_image_fetures, student_text_fetures, 
                            teacher_image_fetures, teacher_text_fetures, 
                            student_config, pid, epsilon=1e-8):
    """
    简称: intra_ts_sdm_loss with hard labels
    """
    batch_size = student_image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    # teacher normalized features for soft labels
    teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
    teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)

    student_image_norm = student_image_fetures / student_image_fetures.norm(dim=1, keepdim=True)
    student_text_norm = student_text_fetures / student_text_fetures.norm(dim=1, keepdim=True)

    logit_scale = 1. / student_config.distillation_intra_tch_stu_sdm_temperature

    stu2tch_t2t_cosine_theta = logit_scale * student_text_norm @ teacher_text_norm.t()
    stu2tch_i2i_cosine_theta = logit_scale * student_image_norm @ teacher_image_norm.t()

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    t2t_pred = F.softmax(stu2tch_t2t_cosine_theta, dim=1)
    t2t_loss = t2t_pred * (F.log_softmax(stu2tch_t2t_cosine_theta, dim=1) - torch.log(labels_distribute + epsilon))

    i2i_pred = F.softmax(stu2tch_i2i_cosine_theta, dim=1)
    i2i_loss = i2i_pred * (F.log_softmax(stu2tch_i2i_cosine_theta, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(t2t_loss, dim=1)) + torch.mean(torch.sum(i2i_loss, dim=1))

    return loss


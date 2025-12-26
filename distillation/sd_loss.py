import torch
import torch.nn.functional as F


def distillation_intra_model_stu_stu_sd_loss(student_image_fetures, student_text_fetures, 
                                            teacher_image_fetures, teacher_text_fetures,
                                            vision_distill=True, text_distill=True):
    """
        简称： intra_ss_sd_loss
    """    
    if vision_distill:         
        teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
        student_image_norm = student_image_fetures / student_image_fetures.norm(dim=-1, keepdim=True)
        student_i2i_sim = student_image_norm @ student_image_norm.t()
        teacher_i2i_sim = teacher_image_norm @ teacher_image_norm.t()
        loss_i = F.mse_loss(student_i2i_sim, teacher_i2i_sim)

    if text_distill:
        teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)   
        student_text_norm = student_text_fetures / student_text_fetures.norm(dim=-1, keepdim=True)
        student_t2t_sim = student_text_norm @ student_text_norm.t()
        teacher_t2t_sim = teacher_text_norm @ teacher_text_norm.t()
        loss_t = F.mse_loss(student_t2t_sim, teacher_t2t_sim)
    
    if vision_distill and not text_distill:
        return loss_i
    elif not vision_distill and text_distill:
        return loss_t
    elif vision_distill and text_distill:
        return (loss_t + loss_i) / 2.
    else:
        raise ValueError("No distillation task is selected")


def distillation_inter_model_stu_stu_sd_loss(student_image_fetures, student_text_fetures, 
                                            teacher_image_fetures, teacher_text_fetures):
    """
        简称： inter_ss_sd_loss
    """    
    # teacher normalized features for soft labels
    teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
    teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)

    # student normalized features
    student_image_norm = student_image_fetures / student_image_fetures.norm(dim=-1, keepdim=True)
    student_text_norm = student_text_fetures / student_text_fetures.norm(dim=-1, keepdim=True)

    student_t2i_sim = student_text_norm @ student_image_norm.t()
    teacher_t2i_sim = teacher_text_norm @ teacher_image_norm.t()

    student_i2t_sim = student_image_norm @ student_text_norm.t()
    teacher_i2t_sim = teacher_image_norm @ teacher_text_norm.t()

    sd_loss = 0.5 * (F.mse_loss(student_t2i_sim, teacher_t2i_sim) + 
                     F.mse_loss(student_i2t_sim, teacher_i2t_sim))
    return sd_loss


def distillation_intra_model_tch_stu_sd_loss(student_image_fetures, student_text_fetures, 
                                            teacher_image_fetures, teacher_text_fetures,
                                            vision_distill=True, text_distill=True):
    """
        简称： intra_ts_sd_loss
    """   
    if vision_distill:
        teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
        student_image_norm = student_image_fetures / student_image_fetures.norm(dim=-1, keepdim=True)
        stu2tch_i2i_sim = student_image_norm @ teacher_image_norm.t()
        tch2tch_i2i_sim = teacher_image_norm @ teacher_image_norm.t()
        loss_i = F.mse_loss(stu2tch_i2i_sim, tch2tch_i2i_sim)

    if text_distill:
        teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)    
        student_text_norm = student_text_fetures / student_text_fetures.norm(dim=-1, keepdim=True)
        stu2tch_t2t_sim = student_text_norm @ teacher_text_norm.t()
        tch2tch_t2t_sim = teacher_text_norm @ teacher_text_norm.t()
        loss_t = F.mse_loss(stu2tch_t2t_sim, tch2tch_t2t_sim)
    
    if vision_distill and not text_distill:
        return loss_i
    elif not vision_distill and text_distill:
        return loss_t
    elif vision_distill and text_distill:
        return (loss_t + loss_i) / 2.
    else:
        raise ValueError("No distillation task is selected")


def distillation_inter_model_tch_stu_sd_loss(student_image_fetures, student_text_fetures, 
                                            teacher_image_fetures, teacher_text_fetures):
    """
        简称： inter_ts_sd_loss
    """    
    # teacher normalized features for soft labels
    teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
    teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)

    # student normalized features
    student_image_norm = student_image_fetures / student_image_fetures.norm(dim=-1, keepdim=True)
    student_text_norm = student_text_fetures / student_text_fetures.norm(dim=-1, keepdim=True)

    stu2tch_t2i_sim = student_text_norm @ teacher_image_norm.t()
    tch2tch_t2i_sim = teacher_text_norm @ teacher_image_norm.t()

    stu2tch_i2t_sim = student_image_norm @ teacher_text_norm.t()
    tch2tch_i2t_sim = teacher_image_norm @ teacher_text_norm.t()

    sd_loss = 0.5 * (F.mse_loss(stu2tch_t2i_sim, tch2tch_t2i_sim) + 
                     F.mse_loss(stu2tch_i2t_sim, tch2tch_i2t_sim))
    return sd_loss


def distillation_intra_model_tch_stu_sym_sd_loss(student_image_fetures, student_text_fetures, 
                                            teacher_image_fetures, teacher_text_fetures):
    """
        简称： intra_ts_sym_sd_loss
    """    
    # teacher normalized features for soft labels
    teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
    teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)

    # student normalized features
    student_image_norm = student_image_fetures / student_image_fetures.norm(dim=-1, keepdim=True)
    student_text_norm = student_text_fetures / student_text_fetures.norm(dim=-1, keepdim=True)

    stu2tch_t2t_sim = student_text_norm @ teacher_text_norm.t()
    stu2tch_i2i_sim = student_image_norm @ teacher_image_norm.t()

    sd_loss = F.mse_loss(stu2tch_t2t_sim, stu2tch_i2i_sim) 

    return sd_loss


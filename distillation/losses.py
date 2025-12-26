# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn import MSELoss, KLDivLoss, CosineEmbeddingLoss
import math
import torch.nn.functional as F


class TransformerLosses():
    """Implements transformer specific loss functions for Knowledge Distillation.
    """

    def __init__(self, student_config, teacher_config, device, args):
        self.mse_loss = MSELoss()
        self.kl_loss = KLDivLoss(reduction='batchmean')
        self.cosine_loss = CosineEmbeddingLoss()
        self.distill_config = student_config.distillation_config
        self.device = device
        self.student_config = student_config
        self.teacher_config = teacher_config
        self.student_num_attention_heads = student_config.vision_width // student_config.vision_heads_size
        self.teacher_num_attention_heads = teacher_config.vision_width // teacher_config.vision_heads_size
        self.batch_size = args.batch_size

    def compute_loss_(self, pred, target, loss_name):

        if self.distill_config[loss_name] == "mse":
            return self.mse_loss(pred, target)

        elif self.distill_config[loss_name] == "kld":
            seq_length = pred.size(0) if loss_name == "value_state_loss" else pred.size(-1)
            if loss_name == "value_state_loss":
                dk_student = pred.shape[-1] // self.student_num_attention_heads
                dk_teacher = target.shape[-1] // self.teacher_num_attention_heads
                # Old: (bsz, seq, heads * dk) => (bsz, heads, seq, dk)
                # New: (seq, bsz, heads * dk) => (bsz * heads, seq, dk)
                student_values = pred.view(seq_length, self.batch_size * self.student_num_attention_heads,
                                           dk_student)
                student_values = student_values.transpose(0, 1)
                teacher_values = target.view(seq_length, self.batch_size * self.teacher_num_attention_heads,
                                             dk_teacher)
                teacher_values = teacher_values.transpose(0, 1)
                # (..., seq, dk) x (..., dk, seq)
                pred = torch.bmm(student_values, student_values.transpose(1, 2)) / math.sqrt(dk_student)
                target = torch.bmm(teacher_values, teacher_values.transpose(1, 2)) / math.sqrt(dk_teacher)

                pred = pred.view(self.batch_size, self.student_num_attention_heads, seq_length, seq_length)
                target = target.view(self.batch_size, self.teacher_num_attention_heads, seq_length, seq_length)

            return self.kl_loss(torch.nn.LogSoftmax(dim=-1)(pred), torch.nn.Softmax(dim=-1)(target)) / (
                        self.student_num_attention_heads * seq_length)

        elif self.distill_config[loss_name] == "cosine":
            # seq_length = pred.size(0)
            # return self.cosine_loss(pred.transpose(0, 2).reshape(-1, seq_length),
            #                         target.transpose(0, 2).reshape(-1, seq_length),
            #                         torch.tensor([1]).to(self.device))
            return self.cosine_loss(pred.view(-1, self.teacher_config.vision_width),
                                    target.view(-1, self.teacher_config.vision_width),
                                    torch.tensor([1]).to(self.device))

        elif self.distill_config[loss_name] == "sdm":
            pass


        else:
            error_string = "'attention_loss':{} not defined. Choose among 'mse', 'cosine' or 'kld'".format(
                self.distill_config["attention_loss"])
            raise ValueError(error_string)

    def compute_loss(self, pred, target, loss_name):

        loss = 0.
        for student, teacher in zip(pred, target):
            if loss_name == "attention_loss":
                # 将小于等于 -1e2 的值替换为 0
                student = torch.where(student <= -1e2, torch.zeros_like(student).to(self.device),
                                      student)
                teacher = torch.where(teacher <= -1e2, torch.zeros_like(teacher).to(self.device),
                                      teacher)
            tmp_loss = self.compute_loss_(student, teacher, loss_name)
            loss += tmp_loss
        return loss

def distillation_fd_loss(student_image_fetures, student_text_fetures, 
                        teacher_image_fetures, teacher_text_fetures):
    """
    SD loss:
        calculate text features mse between student and teacher
        calculate image features mse between student and teacher
    """
    N, d = student_image_fetures.shape[0], student_image_fetures.shape[1]
    # teacher normalized features for soft labels
    teacher_image_norm = teacher_image_fetures / teacher_image_fetures.norm(dim=-1, keepdim=True)
    teacher_text_norm = teacher_text_fetures / teacher_text_fetures.norm(dim=-1, keepdim=True)

    # student normalized features
    student_image_norm = student_image_fetures / student_image_fetures.norm(dim=-1, keepdim=True)
    student_text_norm = student_text_fetures / student_text_fetures.norm(dim=-1, keepdim=True)

    sd_loss = 0.5 * (F.mse_loss(student_image_norm, teacher_image_norm) + 
                     F.mse_loss(student_text_norm, teacher_text_norm))
    return sd_loss
def compute_clip_loss(image_embeds, text_embeds, temperature=0.02):
    """
    计算 CLIP 对比损失
    Args:
        image_embeds: 图像特征 [batch_size, dim]
        text_embeds: 文本特征 [batch_size, dim]
        temperature: 温度参数
    Returns:
        loss: CLIP 损失
    """
    batch_size = image_embeds.shape[0]
    labels = torch.arange(batch_size, device=image_embeds.device)
    
    # 归一化
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    # 计算相似度
    logit_scale = 1.0 / temperature
    logits_per_image = logit_scale * image_embeds @ text_embeds.t()
    logits_per_text = logits_per_image.t()
    
    # 对比损失
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    return (loss_i + loss_t) / 2.0


def feature_distillation_loss(student_image_embeds, teacher_image_embeds,
                              student_text_embeds, teacher_text_embeds,
                              fd_weight=2000.0):
    """
    FD: Feature Distillation Loss
    使用 MSE 损失对齐学生和教师的特征
    
    Args:
        student_image_embeds: 学生图像特征 [batch_size, dim]
        teacher_image_embeds: 教师图像特征 [batch_size, dim]
        student_text_embeds: 学生文本特征 [batch_size, dim]
        teacher_text_embeds: 教师文本特征 [batch_size, dim]
        fd_weight: FD 权重
    Returns:
        loss: FD 损失
    """
    # 归一化特征
    student_image_norm = F.normalize(student_image_embeds, dim=-1)
    teacher_image_norm = F.normalize(teacher_image_embeds, dim=-1)
    student_text_norm = F.normalize(student_text_embeds, dim=-1)
    teacher_text_norm = F.normalize(teacher_text_embeds, dim=-1)
    
    # 计算 MSE 损失
    loss_image = F.mse_loss(student_image_norm, teacher_image_norm)
    loss_text = F.mse_loss(student_text_norm, teacher_text_norm)
    
    # 加权求和
    fd_loss = fd_weight * (loss_image + loss_text) / 2.0
    
    return fd_loss


def interactive_contrastive_learning_loss(student_image_embeds, teacher_image_embeds,
                                         student_text_embeds, teacher_text_embeds,
                                         temperature=0.02, icl_weight=1.0):
    """
    ICL: Interactive Contrastive Learning Loss
    学生图像特征与教师文本特征对比，学生文本特征与教师图像特征对比
    
    Args:
        student_image_embeds: 学生图像特征 [batch_size, dim]
        teacher_image_embeds: 教师图像特征 [batch_size, dim]
        student_text_embeds: 学生文本特征 [batch_size, dim]
        teacher_text_embeds: 教师文本特征 [batch_size, dim]
        temperature: 温度参数
        icl_weight: ICL 权重
    Returns:
        loss: ICL 损失
    """
    batch_size = student_image_embeds.shape[0]
    labels = torch.arange(batch_size, device=student_image_embeds.device)
    
    # 归一化
    student_image_norm = F.normalize(student_image_embeds, dim=-1)
    teacher_image_norm = F.normalize(teacher_image_embeds, dim=-1)
    student_text_norm = F.normalize(student_text_embeds, dim=-1)
    teacher_text_norm = F.normalize(teacher_text_embeds, dim=-1)
    
    logit_scale = 1.0 / temperature
    
    # 学生图像 vs 教师文本
    logits_si_tt = logit_scale * student_image_norm @ teacher_text_norm.t()
    loss_si_tt = F.cross_entropy(logits_si_tt, labels)
    
    # 学生文本 vs 教师图像
    logits_st_ti = logit_scale * student_text_norm @ teacher_image_norm.t()
    loss_st_ti = F.cross_entropy(logits_st_ti, labels)
    
    # 加权求和
    icl_loss = icl_weight * (loss_si_tt + loss_st_ti) / 2.0
    
    return icl_loss


def contrastive_relational_distillation_loss(student_image_embeds, teacher_image_embeds,
                                            student_text_embeds, teacher_text_embeds,
                                            temperature=0.02, crd_weight=1.0):
    """
    CRD: Contrastive Relational Distillation Loss
    使用 KL 散度对齐学生和教师的关系分布
    
    Args:
        student_image_embeds: 学生图像特征 [batch_size, dim]
        teacher_image_embeds: 教师图像特征 [batch_size, dim]
        student_text_embeds: 学生文本特征 [batch_size, dim]
        teacher_text_embeds: 教师文本特征 [batch_size, dim]
        temperature: 温度参数
        crd_weight: CRD 权重
    Returns:
        loss: CRD 损失
    """
    # 归一化
    student_image_norm = F.normalize(student_image_embeds, dim=-1)
    teacher_image_norm = F.normalize(teacher_image_embeds, dim=-1)
    student_text_norm = F.normalize(student_text_embeds, dim=-1)
    teacher_text_norm = F.normalize(teacher_text_embeds, dim=-1)
    
    logit_scale = 1.0 / temperature
    
    # I2T: Image to Text 关系
    student_i2t = logit_scale * student_image_norm @ student_text_norm.t()
    teacher_i2t = logit_scale * teacher_image_norm @ teacher_text_norm.t()
    
    student_i2t_prob = F.log_softmax(student_i2t, dim=-1)
    teacher_i2t_prob = F.softmax(teacher_i2t, dim=-1)
    loss_i2t = F.kl_div(student_i2t_prob, teacher_i2t_prob, reduction='batchmean')
    
    # T2I: Text to Image 关系
    student_t2i = logit_scale * student_text_norm @ student_image_norm.t()
    teacher_t2i = logit_scale * teacher_text_norm @ teacher_image_norm.t()
    
    student_t2i_prob = F.log_softmax(student_t2i, dim=-1)
    teacher_t2i_prob = F.softmax(teacher_t2i, dim=-1)
    loss_t2i = F.kl_div(student_t2i_prob, teacher_t2i_prob, reduction='batchmean')
    
    # 加权求和
    crd_loss = crd_weight * (loss_i2t + loss_t2i) / 2.0
    
    return crd_loss


def compute_clipkd_loss(student_image_embeds, teacher_image_embeds,
                       student_text_embeds, teacher_text_embeds,
                       temperature=0.02,
                       use_fd=True, use_icl=True, use_crd=True,
                       fd_weight=2000.0, icl_weight=1.0, crd_weight=1.0,
                       return_details=False):
    """
    计算完整的 CLIP-KD 损失
    
    Args:
        student_image_embeds: 学生图像特征
        teacher_image_embeds: 教师图像特征
        student_text_embeds: 学生文本特征
        teacher_text_embeds: 教师文本特征
        temperature: 温度参数
        use_fd: 是否使用 FD 损失
        use_icl: 是否使用 ICL 损失
        use_crd: 是否使用 CRD 损失
        fd_weight: FD 权重
        icl_weight: ICL 权重
        crd_weight: CRD 权重
        return_details: 是否返回详细损失
    
    Returns:
        如果 return_details=False: 返回总损失
        如果 return_details=True: 返回 (总损失, 损失字典)
    """
    total_loss = 0.0
    loss_dict = {}
    
    # Feature Distillation
    if use_fd:
        fd_loss = feature_distillation_loss(
            student_image_embeds, teacher_image_embeds,
            student_text_embeds, teacher_text_embeds,
            fd_weight=fd_weight
        )
        total_loss += fd_loss
        loss_dict['fd_loss'] = fd_loss.item()
    
    # Interactive Contrastive Learning
    if use_icl:
        icl_loss = interactive_contrastive_learning_loss(
            student_image_embeds, teacher_image_embeds,
            student_text_embeds, teacher_text_embeds,
            temperature=temperature,
            icl_weight=icl_weight
        )
        total_loss += icl_loss
        loss_dict['icl_loss'] = icl_loss.item()
    
    # Contrastive Relational Distillation
    if use_crd:
        crd_loss = contrastive_relational_distillation_loss(
            student_image_embeds, teacher_image_embeds,
            student_text_embeds, teacher_text_embeds,
            temperature=temperature,
            crd_weight=crd_weight
        )
        total_loss += crd_loss
        loss_dict['crd_loss'] = crd_loss.item()
    
    if return_details:
        return total_loss, loss_dict
    return total_loss

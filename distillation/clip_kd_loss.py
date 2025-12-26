"""
CLIP-KD Loss Implementation
Based on "CLIP-KD: An Empirical Study of CLIP Model Distillation" (CVPR 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPKDLoss(nn.Module):  
    """
    CLIP-KD: Combines FD (Feature Distillation), ICL (Interactive Contrastive Learning),
    and CRD (Contrastive Relational Distillation)
    """
    def __init__(self, temperature=0.07, lambda_crd=1.0, lambda_fd=2000.0, lambda_icl=1.0):  
        super().__init__()
        self.temperature = temperature
        self.lambda_crd = lambda_crd
        self.lambda_fd = lambda_fd
        self.lambda_icl = lambda_icl
        
    def forward(self, student_image_features, student_text_features,
                teacher_image_features, teacher_text_features,
                use_fd=True, use_icl=True, use_crd=True):  
        """
        Args:
            student_image_features: [batch_size, dim]
            student_text_features: [batch_size, dim]
            teacher_image_features: [batch_size, dim]
            teacher_text_features: [batch_size, dim]
            use_fd: 是否使用FD
            use_icl: 是否使用ICL
            use_crd: 是否使用CRD
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 1. Feature Distillation (FD) - 论文公式(11)
        if use_fd:
            fd_loss = self.feature_distillation(
                student_image_features, student_text_features,
                teacher_image_features, teacher_text_features
            )
            total_loss += self.lambda_fd * fd_loss
            loss_dict['fd_loss'] = fd_loss.item()
        
        # 2. Interactive Contrastive Learning (ICL) - 论文公式(14)(15)(16)
        if use_icl:
            icl_loss = self.interactive_contrastive_learning(
                student_image_features, student_text_features,
                teacher_image_features, teacher_text_features
            )
            total_loss += self.lambda_icl * icl_loss
            loss_dict['icl_loss'] = icl_loss.item()
        
        # 3. Contrastive Relational Distillation (CRD) - 论文公式(8)(9)(10)
        if use_crd:
            crd_loss = self.contrastive_relational_distillation(
                student_image_features, student_text_features,
                teacher_image_features, teacher_text_features
            )
            total_loss += self.lambda_crd * crd_loss
            loss_dict['crd_loss'] = crd_loss.item()
        
        # ✅ 添加total_loss到字典
        loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict
    
    def feature_distillation(self, student_image_features, student_text_features,
                            teacher_image_features, teacher_text_features):
        """
        Feature Distillation (FD) - 公式(11)
        L_FD = 1/|B| * Σ(||v_k^T - v_k^S||_2^2 + ||s_k^T - s_k^S||_2^2)
        """
        # L2归一化
        student_image_norm = F.normalize(student_image_features, p=2, dim=-1)
        student_text_norm = F.normalize(student_text_features, p=2, dim=-1)
        teacher_image_norm = F.normalize(teacher_image_features, p=2, dim=-1)
        teacher_text_norm = F.normalize(teacher_text_features, p=2, dim=-1)
        
        # MSE损失
        image_loss = F.mse_loss(student_image_norm, teacher_image_norm)
        text_loss = F.mse_loss(student_text_norm, teacher_text_norm)
        
        return image_loss + text_loss
    
    def interactive_contrastive_learning(self, student_image_features, student_text_features,
                                        teacher_image_features, teacher_text_features):
        """
        Interactive Contrastive Learning (ICL) - 公式(14)(15)(16)
        学生作为anchor，与teacher的embeddings进行对比学习
        """
        # L2归一化
        student_image_norm = F.normalize(student_image_features, p=2, dim=-1)
        student_text_norm = F.normalize(student_text_features, p=2, dim=-1)
        teacher_image_norm = F.normalize(teacher_image_features, p=2, dim=-1)
        teacher_text_norm = F.normalize(teacher_text_features, p=2, dim=-1)
        
        # 计算相似度矩阵
        # student image与teacher text的相似度
        i2t_logits = student_image_norm @ teacher_text_norm.t() / self.temperature
        # student text与teacher image的相似度
        t2i_logits = student_text_norm @ teacher_image_norm.t() / self.temperature
        
        batch_size = student_image_features.size(0)
        labels = torch.arange(batch_size, device=student_image_features.device)
        
        # 对比学习损失
        i2t_loss = F.cross_entropy(i2t_logits, labels)
        t2i_loss = F.cross_entropy(t2i_logits, labels)
        
        return (i2t_loss + t2i_loss) / 2.0
    
    def contrastive_relational_distillation(self, student_image_features, student_text_features,
                                           teacher_image_features, teacher_text_features):
        """
        Contrastive Relational Distillation (CRD) - 公式(8)(9)(10)
        对齐teacher和student的对比分布
        """
        # L2归一化
        student_image_norm = F.normalize(student_image_features, p=2, dim=-1)
        student_text_norm = F.normalize(student_text_features, p=2, dim=-1)
        teacher_image_norm = F.normalize(teacher_image_features, p=2, dim=-1)
        teacher_text_norm = F.normalize(teacher_text_features, p=2, dim=-1)
        
        # Teacher的对比分布
        teacher_i2t_logits = teacher_image_norm @ teacher_text_norm.t() / self.temperature
        teacher_t2i_logits = teacher_text_norm @ teacher_image_norm.t() / self.temperature
        
        teacher_i2t_prob = F.softmax(teacher_i2t_logits, dim=1)
        teacher_t2i_prob = F.softmax(teacher_t2i_logits, dim=1)
        
        # Student的对比分布
        student_i2t_logits = student_image_norm @ student_text_norm.t() / self.temperature
        student_t2i_logits = student_text_norm @ student_image_norm.t() / self.temperature
        
        student_i2t_log_prob = F.log_softmax(student_i2t_logits, dim=1)
        student_t2i_log_prob = F.log_softmax(student_t2i_logits, dim=1)
        
        # KL散度
        i2t_loss = F.kl_div(student_i2t_log_prob, teacher_i2t_prob, reduction='batchmean')
        t2i_loss = F.kl_div(student_t2i_log_prob, teacher_t2i_prob, reduction='batchmean')
        
        return i2t_loss + t2i_loss


def compute_clip_loss(image_features, text_features, temperature):
    """
    计算CLIP的原始对比损失（用于任务损失）
    
    Args:
        image_features: [batch_size, dim]
        text_features: [batch_size, dim]
        temperature: float
    """
    # L2归一化
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    
    # 计算logits
    logits_per_image = image_features @ text_features.t() / temperature
    logits_per_text = logits_per_image.t()
    
    batch_size = image_features.size(0)
    labels = torch.arange(batch_size, device=image_features.device)
    
    # 对比损失
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    
    return (loss_i2t + loss_t2i) / 2.0

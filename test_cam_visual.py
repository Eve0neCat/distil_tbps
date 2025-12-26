'''
Author: gengyou.lu 1770591868@qq.com
Date: 2025-03-06 15:37:38
FilePath: /IRRA/test_cam_visual.py
LastEditTime: 2025-03-08 15:08:14
Description: 
'''
from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


# 新导入库
import json
import torch.nn.functional as F
from datasets.build import build_transforms
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from datasets.bases import tokenize
from utils.metrics import rank


# 可视化相关库
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization
from torchvision import transforms
from IPython.display import HTML


import torch_npu
from torch_npu.contrib import transfer_to_npu
from matplotlib.colors import LinearSegmentedColormap

#image Transformer的层数
start_layer =  -1
#text Transformer的层数
start_layer_text =  -1

#获得注意力机制热力图结果
def interpret(image, texts, model, device, start_layer=start_layer, start_layer_text=start_layer_text):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)#将图像张量 image 复制 batch_size 次，以匹配文本输入的批次大小
    # logits_per_image, logits_per_text = model(images, texts)
    img_feat = model.encode_image(img)
    img_feat = F.normalize(img_feat, p=2, dim=1) # 归一化 image features
    text_feat = model.encode_text(caption)    
    text_feat = F.normalize(text_feat, p=2, dim=1)
        
    logits_per_image =  img_feat @ text_feat.t()
    logits_per_text = text_feat @ img_feat.t()

    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()#计算每个图像与文本对的概率分布
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)#初始化，创建一个形状为(batch_size, batch_size)的零矩阵，用于存储one-hot编码
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1#将对角线位置设置为1，其余位置保持为0。
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)#将 NumPy 数组转换为 PyTorch 张量，并启用梯度计算，以便在后续的反向传播中计算梯度。
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)#与模型的输出相乘，选择每个图像与对应文本的相似度分数，计算所有选择的相似度分数的总和。
    #将图像文本对输入CLIP模型后,拿到one-hot输出，作为各个类别标签
    model.zero_grad()

    image_attn_blocks = list(dict(model.base_model.visual.transformer.resblocks.named_children()).values())#12个ResidualAttentionBlock

    if start_layer == -1:
      # calculate index of last layer
      start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]#获取第一个图像注意力块的注意力概率矩阵的最后一个维度的大小，即每个图像被分割成的 Patch 数量。
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)#将 R 扩展到形状 [batch_size, num_tokens, num_tokens]，即对每个样本都有一个独立的单位矩阵。
    for i, blk in enumerate(image_attn_blocks):#遍历图像部分的每个注意力块
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()#计算 one_hot 对当前块的注意力概率 blk.attn_probs 的梯度。
        cam = blk.attn_probs.detach()#提取当前块的注意力概率
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])#[batch_size, num_heads, num_tokens, num_tokens]
        cam = cam.clamp(min=0).mean(dim=1)#将相关性分数中的负值设置为 0，求均值。[batch_size, num_tokens, num_tokens]
        R = R + torch.bmm(cam, R)#更新相关性矩阵 R
    image_relevance = R[:, 0, 1:]
    #将相关性分数可视化为热力图，叠加在原始图像上，直观地显示模型关注的区域。

    #类似上部分，计算文本中对各词注意力大小
    text_attn_blocks = list(dict(model.base_model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1:
      # calculate index of last layer
      start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text

    return text_relevance, image_relevance

def show_image_relevance(image_relevance, image, orig_image):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)#applyColorMap将相关性分数转换为彩色热力图
        heatmap = 0.8 * np.float32(heatmap) / 255 # 0.8可在[0,1]中修改，调整热力图透明度
        cam = heatmap + np.float32(img)#将热力图与原始图像叠加
        cam = cam / np.max(cam)
        return cam

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    axs[0].imshow(orig_image)
    axs[0].axis('off')

    # dim = int(image_relevance.numel() ** 0.5)
    # image_relevance = image_relevance.reshape(1, 1, dim, dim)
    # image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    # image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    # 修改
    image_relevance = image_relevance.reshape(1, 1, 24, 8)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=(384, 128), mode='bilinear')
    image_relevance = image_relevance.reshape(384, 128).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    #将相关性分数调整为与原始图像相同的大小，并归一化。
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)#将相关性分数叠加到原始图像上，并显示叠加后的图像。
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    axs[1].imshow(vis)
    axs[1].axis('off')


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def single_show_image_relevance(image_relevance, image, orig_image):
    # create heatmap from mask on image
    # def show_cam_on_image(img, mask):
    #     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)#applyColorMap将相关性分数转换为彩色热力图        
    #     heatmap = 0.8 * np.float32(heatmap) / 255 # 0.8可在[0,1]中修改，调整热力图透明度
    #     cam = heatmap + np.float32(img)#将热力图与原始图像叠加
    #     cam = cam / np.max(cam)
    #     return cam    

    image_relevance = image_relevance.reshape(1, 1, 24, 8)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=(384, 128), mode='bilinear')
    image_relevance = image_relevance.reshape(384, 128).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    #将相关性分数调整为与原始图像相同的大小，并归一化。
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    # vis = show_cam_on_image(image, image_relevance)#将相关性分数叠加到原始图像上，并显示叠加后的图像。
    # vis = np.uint8(255 * vis)
    vis = show_cam_on_image(image, image_relevance, image_weight=0.5)
    return vis


def show_heatmap_on_text(text, text_encoding, R_text):
    CLS_idx = text_encoding.argmax(dim=-1)
    R_text = R_text[CLS_idx, 1:CLS_idx]
    text_scores = R_text / R_text.sum()
    # text_scores = text_scores.flatten()
    text_scores = text_scores.flatten().detach().cpu().numpy()
    print(text_scores)
    tokenizer = SimpleTokenizer() 
    text_tokens = tokenizer.encode(text)
    text_tokens_decoded = [tokenizer.decode([a]) for a in text_tokens]
    vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,text_tokens_decoded,1)]
    visualization.visualize_text(vis_data_records)


    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 9  # 设置全局字体大小

    # 设置图像尺寸与 A4 纸等宽
    a4_width_inch = 210 / 25.4  # A4 纸宽度转换为英寸
    fig_height_inch = 2  # 保持高度不变

    # 创建热力图
    fig, ax = plt.subplots(figsize=(a4_width_inch, fig_height_inch))

    # 创建热力图
    # fig, ax = plt.subplots(figsize=(10, 2))    
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", "yellow", "red"])
    heatmap = ax.pcolor(np.array([text_scores]), cmap=cmap, vmin=0, vmax=1)

    # 设置x轴为文本tokens
    ax.set_xticks(np.arange(len(text_tokens_decoded)) + 0.5, minor=False)
    ax.set_xticklabels(text_tokens_decoded, rotation=45, ha="right", fontsize=9)

    # 设置y轴不可见
    ax.get_yaxis().set_visible(False)

    # 添加颜色条
    # plt.colorbar(heatmap)
    cbar = plt.colorbar(heatmap, pad=0.01)  # 缩小 colorbar 与图的间距
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(['0.0', '0.5', '1.0'])
    cbar.ax.tick_params(labelsize=9)  # 设置 colorbar 的字体大小

    # 保存图像
    plt.tight_layout()
    # plt.savefig(f'./{test_model}_text_heatmap.png', bbox_inches='tight', dpi=300)
    # plt.close()

if __name__ == '__main__':

    # test_model = "tch" 
    # test_model = "stu"  
    test_model = "MoTIS"  

    parser = argparse.ArgumentParser(description="IRRA Test")
    # parser.add_argument("--config_file", default='logs/CUHK-PEDES/baseline/configs.yaml')
    if test_model == "stu":
        parser.add_argument("--config_file", default='logs/CUHK-PEDES/20250124_150702_tinyclip_yfcc15m_xsmall-dist_ours_base_MoTIS_loss2/configs.yaml')
    elif test_model == "tch":
        parser.add_argument("--config_file", default='logs/CUHK-PEDES/20241202_201442_iira/configs.yaml')  
    elif test_model == "MoTIS":
        parser.add_argument("--config_file", default='logs/CUHK-PEDES/20250120_215726_sequence_tinyclip_yfcc15m_xsmall-dist_MoTIS-task_itc/configs.yaml')
          
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    args.training = False
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    # do_inference(model, test_img_loader, test_txt_loader)

    dataset_name = "CUHK-PEDES"
    path = "CUHK-PEDES/imgs/CUHK03/564_3.jpg"
    query = "A teenage girl with long black hair wearing a navy blue hoodie with a white undershirt and brown pants with white sneakers and a white backpack."      

    # dataset_name = "ICFG-PEDES"
    # path = "ICFG-PEDES/imgs/test/1712/1712_023_05_0302morning_0312_1.jpg"
    # query = "A young man with short black hair is wearing a hooded grey-blue patched parachute jacket and faded blue denim jeans. He is also wearing brown loafers and is carrying a black backpack strapped to his shoulders."

    # dataset_name = "RSTPReid" 
    # path = "RSTPReid/imgs/3891_c15_0016.jpg"
    # query = "A woman  is wearing a yellow down jacket,black trousers and a pair of black boots.She is also carrying black bag over her left shoulder."

    save_path = f"./heapmap_result/{dataset_name}"    
    root_data = args.root_dir    
    # root_data = "./"
    # 进行图像编码
    img_preprocess = build_transforms(img_size=(384, 128), is_train=False)
    img_path = os.path.join(root_data, path)
    img = read_image(img_path)
    img = img_preprocess(img).unsqueeze(0)
    img = img.to(device)

    # 进行文本编码
    tokenizer = SimpleTokenizer()        
    text = tokenize(query, tokenizer=tokenizer, text_length=77, truncate=True).unsqueeze(0)
    caption = text.to(device)  
    
    R_text, R_image = interpret(model=model, image=img, texts=text, device=device)    
    batch_size = text.shape[0]
    for i in range(batch_size):        
        show_heatmap_on_text(query, text, R_text[i])
        plt.savefig(f'{save_path}/{test_model}_text_heatmap.png', bbox_inches='tight', dpi=300)
        plt.close()
        orig_image = Image.open(img_path)
        orig_image = orig_image.resize((128, 384))
        vis = single_show_image_relevance(R_image[i], img, orig_image) 
        cv2.imwrite(f'{save_path}/{test_model}_image_heatmap.png', vis)


    ''' 文本检索图像 '''
    # save_path = "./heapmap_result/CUHK-PEDES/94"
    # for ii in range(10):
    #     root_data = args.root_dir    
    #     # 进行图像编码
    #     img_preprocess = build_transforms(img_size=(384, 128), is_train=False)
    #     img_path = os.path.join(path, f"top_{ii}.jpg")
    #     img = read_image(img_path)
    #     img = img_preprocess(img).unsqueeze(0)
    #     img = img.to(device)

    #     # 进行文本编码
    #     tokenizer = SimpleTokenizer()        
    #     text = tokenize(query, tokenizer=tokenizer, text_length=77, truncate=True).unsqueeze(0)
    #     caption = text.to(device)  
        
    #     R_text, R_image = interpret(model=model, image=img, texts=text, device=device)    
    #     batch_size = text.shape[0]
    #     for i in range(batch_size):        
    #         show_heatmap_on_text(query, text, R_text[i])
    #         plt.savefig(f'{save_path}/{test_model}_text_heatmap_{ii}.png', bbox_inches='tight', dpi=300)
    #         plt.close()
    #         orig_image = Image.open(img_path)
    #         orig_image = orig_image.resize((128, 384))
    #         vis = single_show_image_relevance(R_image[i], img, orig_image) 
    #         cv2.imwrite(f'{save_path}/{test_model}_image_heatmap_{ii}.png', vis)
            

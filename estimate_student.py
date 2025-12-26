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
from model.build import build_model, build_student_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs

import torch_npu
from torch_npu.contrib import transfer_to_npu

# 新导入库
import json
import torch.nn.functional as F
from datasets.build import build_transforms
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from datasets.bases import tokenize
from utils.metrics import rank

def calculate_model_size(model):
    model_size = sum(p.numel() for p in model.parameters())
    return model_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA Test")
    # parser.add_argument("--config_file", default='logs/CUHK-PEDES/20241230_154620_tinyclip_yfcc15m_xsmall-task_itc/configs.yaml')        
    # parser.add_argument("--config_file", default='logs/CUHK-PEDES/20241230_173111_tinyclip_yfcc15m_xsmall-dist_cross_kd_loss-task_itc/configs.yaml')
    # parser.add_argument("--config_file", default='logs/CUHK-PEDES/20250120_215726_sequence_tinyclip_yfcc15m_xsmall-dist_MoTIS-task_itc/configs.yaml')
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/20250102_144159_sequence_tinyclip_yfcc15m_xsmall-dist_ConaCLIP-task_itc/configs.yaml')    
    # parser.add_argument("--config_file", default='logs/CUHK-PEDES/20250124_150702_tinyclip_yfcc15m_xsmall-dist_ours_base_MoTIS_loss2/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    args.training = False
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    # device = "cuda"
    device = args.device
    torch.cuda.set_device(device)

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    # model = build_model(args, num_classes=num_classes)
    model, stu_config = build_student_model(args, convert_fp16=True)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    # do_inference(model, test_img_loader, test_txt_loader)
    
    ''' 1.PyTorch推理精度测试 '''
    precise_test = True
    if precise_test:
        # # 读取测试数据集 json 文件
        # root_data = args.root_dir
        # test_data_name = "CUHK-PEDES" 
        # test_dataset = json.load(open(os.path.join(root_data, f"{test_data_name}_test_data.json")))
        # image_paths = test_dataset["img_paths"]
        # gids = test_dataset["image_pids"]
        # gids = torch.tensor(gids)
        # Query = test_dataset["captions"]
        # qids = test_dataset["caption_pids"]
        # qids = torch.tensor(qids)
        
        # 自构建数据集
        root_data = args.root_dir
        test_data_name = "UESTC-PEDES" 
        test_dataset = json.load(open(os.path.join(root_data, "UESTC-PEDES/uestc_tbps_dataset.json")))
        image_paths = test_dataset["image_paths"]
        gids = test_dataset["image_pids"]
        gids = torch.tensor(gids)
        Query = test_dataset["captions"]
        qids = test_dataset["caption_pids"]
        qids = torch.tensor(qids)


        # 进行图像编码
        img_preprocess = build_transforms(img_size=(384, 128), is_train=False)
        gfeats = []
        for path in image_paths:
            img_path = os.path.join(root_data, path)
            img = read_image(img_path)
            img = img_preprocess(img).unsqueeze(0)
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gfeats.append(img_feat)
        gfeats = torch.cat(gfeats, 0)
        gfeats = F.normalize(gfeats, p=2, dim=1) # 归一化 image features
        print(gfeats.shape)

        # 进行文本编码
        tokenizer = SimpleTokenizer()                
        qfeats = []
        for text in Query:        
            caption = tokenize(text, tokenizer=tokenizer, text_length=77, truncate=True).unsqueeze(0)
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)        
            qfeats.append(text_feat)        
        qfeats = torch.cat(qfeats, 0)
        qfeats = F.normalize(qfeats, p=2, dim=1) # 归一化 text features
        print(qfeats.shape)

        # 计算相似度
        similarity = qfeats @ gfeats.t()

        # 计算评价指标
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])
        print('\n' + str(table))



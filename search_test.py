import os
import json

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image

# TODO: 导入库
from simple_tokenizer import SimpleTokenizer


''' 图像预处理代码 '''
def build_transforms(img_size=(384, 128)):
    # 图像预处理
    height, width = img_size
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),])    
    return transform

def transfer_pic(input_path, use_aipp=False):
    # 图像预处理
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    mean = np.expand_dims(mean, axis=(1, 2)).astype(np.float32)
    std = np.expand_dims(std, axis=(1, 2)).astype(np.float32)

    with Image.open(input_path) as image_file:
        # img = image_file.convert('RGB')
        # img = img.resize((128, 384)) # 有精度损失
        # img = img.resize((128, 384), Image.BILINEAR)                
        img = image_file.resize((128, 384), Image.BILINEAR)                
    
    if use_aipp is False:
        img = np.array(img).astype(np.float32) / 255.0
        # HWC -> CHW 
        img = img.transpose((2, 0, 1))      
        img = (img - mean) / std          
    else:
        img = np.array(img)

    return np.array([img]) 


''' 文本预处理 '''
def tokenize(caption: str, tokenizer, text_length=77, truncate=True):
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = np.zeros(text_length, dtype=np.int64)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = tokens
    return result


if "__name__" == "__name__":

    # TODO: 加载模型，获得 image_encoder 和 

    result_root = "/home/lugengyou/workspace/Codes/lugy_research/distil_tbps/deploy/CUHK-PEDES/result"

    ''' 制作归一化的测试图像模型推理特征库'''
    # 加载测试图像路径
    test_data_name = "CUHK-PEDES"  
    #  TODO: 修改数据集路径
    root_data = "/home/lugengyou/workspace/Codes/lugy_research/distil_tbps/data" 
    test_dataset = json.load(open(os.path.join(root_data, f"{test_data_name}_test_data.json")))
    image_paths = test_dataset["img_paths"]

    # TODO: 加载图像编码器模型 image_encoder


    image_features = []
    for path in image_paths:
        sample_path = os.path.join(root_data, path)
        img = transfer_pic(sample_path)
        
        # TODO: 修改图像编码器 forward
        result = image_encoder.forward(img)

        result = result[0, :].reshape(1, -1)
        # 进行 L2 归一化
        result = result / np.linalg.norm(result, ord=2, axis=-1, keepdims=True)
        image_features.append(result)

    # 合并结果为 (N, 512) 的矩阵
    image_features = np.concatenate(image_features, axis=0)
    print(image_features.shape)


    # 2. 初始化文本编码器模型
    bpe_path = os.path.join(root_data, "bpe_simple_vocab_16e6.txt.gz")
    tokenizer = SimpleTokenizer(bpe_path)

    # 随机挑选100个文本进行测试
    test_captions = test_dataset["captions"]
    text_len = len(test_captions)
    test_indices = np.random.choice(text_len, 100, replace=False)
    Query = [test_captions[i] for i in test_indices]

    for ii, text in enumerate(Query):
        query = text
        # 4. 接受文本输入，进行文本编码 
        # text = "The man has short, dark hair and wears khaki pants with an oversized grey hoodie. His black backpack hangs from one shoulder."
        text = tokenize(text, tokenizer=tokenizer, text_length=77, truncate=True)
        text = text.reshape((1, 77))
        
        # TODO:
        result = text_encoder.text_forward(text) # npu 计算     
        text_feature = result[text.argmax(axis=-1), :] # 获取最大值的索引对应的特征，即为文本的 cls 特征

            # 3. numpy 计算输出结果
        # 图像特征提前进行 l2 归一化，文本特征是模型输出需要再 similarity model 中进行 l2 归一化
        l2_norm = np.linalg.norm(text_feature, ord=2, axis=1, keepdims=True)
        np_texts = text_feature / l2_norm
        np_texts = np_texts.transpose(1, 0)
        np_sim = np.dot(image_features, np_texts).reshape(1, -1)
        np_sim = np_sim.reshape(1, -1)
        K = 10
        sorted_indices = np.argsort(np_sim, axis=1)[:, ::-1]
        top10_indices = sorted_indices[:, :K]
        top10_values = np.take_along_axis(np_sim, top10_indices, axis=1)
        print("numpy similarity output:")
        print("similarity:\n", top10_values)
        print("index:\n", top10_indices)

        # 保存结果
        save_folder = os.path.join(result_root, f"test_tch_{ii}")

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        result = {"query": query,
                    "top10_values": top10_values.tolist(), 
                    "top10_indices": top10_indices.tolist()}
        json.dump(result, open(os.path.join(save_folder, f"result_{ii}.json"), "w"), indent=4)

        # 将对应图像保存到文件夹
        import shutil
        for i in range(10):
            image_path = image_paths[top10_indices[0, i]]        
            shutil.copy(os.path.join(root_data, image_path), os.path.join(save_folder, f"top_{i}.jpg"))


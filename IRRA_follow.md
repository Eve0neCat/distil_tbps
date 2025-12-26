# NPU 上复现 IRRA 命令
```
python train.py \
--name iira \
--img_aug \
--batch_size 64 \
--MLM \
--loss_names 'sdm+mlm+id' \
--dataset_name 'CUHK-PEDES' \
--root_dir './data' \
--num_epoch 60 \
--device "npu:7" \
>./logs/CUHK-PEDES/training.log 2>&1 &

python train.py \
--name iira \
--img_aug \
--batch_size 64 \
--MLM \
--loss_names 'sdm+mlm+id' \
--dataset_name 'ICFG-PEDES' \
--root_dir './data' \
--num_epoch 60 \
--device "npu:6" \
--test_device "npu:3" \
>./logs/ICFG-PEDES/training.log 2>&1 &

python train.py \
--name iira \
--img_aug \
--batch_size 64 \
--MLM \
--loss_names 'sdm+mlm+id' \
--dataset_name 'RSTPReid' \
--root_dir './data' \
--num_epoch 60 \
--device "npu:5" \
>./logs/RSTPReid/training.log 2>&1 &

```

# 使用知识蒸馏框架训练
# IRRA crop model，vision 12 text 6 layer, task_sdm
```
python distillation/distillation_train.py \
--name irra_crop_V12_T6_model-task_sdm \
--img_aug \
--batch_size 64 \
--dataset_name 'CUHK-PEDES' \
--root_dir './data' \
--num_epoch 60 \
--device "npu:1" \
--distillation \
--distillation_config_path "distillation/CLIP_V12L_T6L_sdm.json" \
>./logs/CUHK-PEDES/training1.log 2>&1 &

```

# 使用知识蒸馏框架训练
# IRRA crop model，vision 6 text 12 layer, task_sdm
```
python distillation/distillation_train.py \
--name irra_crop_V6_T12_model-task_sdm \
--img_aug \
--batch_size 64 \
--dataset_name 'CUHK-PEDES' \
--root_dir './data' \
--num_epoch 60 \
--device "npu:1" \
--distillation \
--distillation_config_path "distillation/CLIP_V6L_T12L_sdm.json" \
>./logs/CUHK-PEDES/training1.log 2>&1 &

```

# IRRA crop model，vision 12 text 6 layer, task_sdm
```
python distillation/distillation_train.py \
--name irra_crop_V12-w384_T6-w512_model-task_sdm \
--img_aug \
--batch_size 64 \
--dataset_name 'CUHK-PEDES' \
--root_dir './data' \
--num_epoch 120 \
--device "npu:2" \
--vision_width 384 \
--vision_from_scratch \
--transformer_layers 6 \
>./logs/CUHK-PEDES/training2.log 2>&1 &

```

# Cross-model KD: dist_cross_kd_loss-task_itc
# 使用教师模型 softmax 的软标签对学生模型进行知识蒸馏
* 单阶段知识蒸馏， Hinton KD
```
python distillation/distillation_train.py \
--name tinyclip_yfcc15m_xsmall-dist_cross_kd_loss-task_itc \
--img_aug \
--batch_size 64 \
--dataset_name 'RSTPReid' \
--root_dir './data' \
--num_epoch 60 \
--device "npu:1" \
--distillation \
--temperature 0.02 \
--distillation_cross_kd_temperature 0.5 \
--distillation_config_path "distillation/distillation_xsmall/ditillation_tinyclip_yfcc15m_xsmall_cross_kd_loss.json" \
--teacher_model_config "logs/RSTPReid/20241202_201600_iira/configs.yaml" \
--pretrain_student_choice 'state_dict_pretrain_model/tinyclip_yfcc15m_xsmall.pth' \
>./logs/RSTPReid/training4.log 2>&1 &

```


# tinyclip_yfcc15m_xsmall, 顺序蒸馏 dist_MoTIS-task_itc
* 两阶段蒸馏，复现 MoTIS 论文结果:固定其中一个编码器，先蒸馏学生视觉编码器，再蒸馏文本编码器
```
python distillation/sequence_distillation_train.py \
--name sequence_tinyclip_yfcc15m_xsmall-dist_MoTIS-task_itc \
--img_aug \
--batch_size 64 \
--dataset_name 'RSTPReid' \
--root_dir './data' \
--num_epoch 60 \
--device "npu:3" \
--distillation \
--sequence_distillation_2stages \
--distillation_MoTIS_temperature 0.5 \
--distillation_inter_ts_kl_temperature 0.5 \
--distillation_inter_ss_kl_temperature 0.5 \
--distillation_config_path "distillation/distillation_xsmall/sequence_ditillation_tinyclip_yfcc15m_xsmall_MoTIS_loss1.json" \
--teacher_model_config "logs/RSTPReid/20241202_201600_iira/configs.yaml" \
--pretrain_student_choice 'state_dict_pretrain_model/tinyclip_yfcc15m_xsmall.pth' \
>./logs/RSTPReid/training3.log 2>&1 &

```

# tinyclip_yfcc15m_xsmall, 顺序蒸馏 dist_MoTIS-task_itc
* 两阶段蒸馏，复现 ConaCLIP 论文结果:固定其中一个编码器，先蒸馏学生视觉编码器，再蒸馏文本编码器
```
python distillation/sequence_distillation_train.py \
--name sequence_tinyclip_yfcc15m_xsmall-dist_ConaCLIP-task_itc \
--img_aug \
--batch_size 64 \
--dataset_name 'CUHK-PEDES' \
--root_dir './data' \
--num_epoch 60 \
--device "npu:6" \
--distillation \
--sequence_distillation_2stages \
--distillation_MoTIS_temperature 0.5 \
--distillation_inter_ts_sym_kl_temperature 0.5 \
--distillation_config_path "distillation/distillation_xsmall/sequence_ditillation_tinyclip_yfcc15m_xsmall_ConaCLIP_loss.json" \
--teacher_model_config "logs/CUHK-PEDES/20241202_201442_iira/configs.yaml" \
--pretrain_student_choice 'state_dict_pretrain_model/tinyclip_yfcc15m_xsmall.pth' \
>./logs/CUHK-PEDES/training6.log 2>&1 &

```

# tinyclip_yfcc15m_xsmall, 顺序蒸馏 dist_ours-task_itc
* 三阶段蒸馏: 先蒸馏图像编码器、再蒸馏文本编码器、最后联合蒸馏整个模型
```
python distillation/sequence_distillation_train.py \
--name sequence_tinyclip_yfcc15m_xsmall-dist_ours-task_itc \
--img_aug \
--batch_size 64 \
--dataset_name 'RSTPReid' \
--root_dir './data' \
--num_epoch 60 \
--device "npu:7" \
--distillation \
--sequence_distillation_3stages \
--temperature 0.02 \
--distillation_MoTIS_temperature 0.5 \
--distillation_inter_ts_kl_temperature 0.5 \
--distillation_inter_ss_kl_temperature 0.5 \
--distillation_inter_ts_sym_kl_temperature 0.5 \
--distillation_config_path "distillation/distillation_xsmall/sequence_ditillation_tinyclip_yfcc15m_xsmall_ours_loss1.json" \
--teacher_model_config "logs/RSTPReid/20241202_201600_iira/configs.yaml" \
--pretrain_student_choice 'state_dict_pretrain_model/tinyclip_yfcc15m_xsmall.pth' \
>./logs/RSTPReid/training7.log 2>&1 &

```


# 三阶段知识蒸馏，基于 MoTIS 两阶段知识蒸馏结果，使用一阶段蒸馏框架继续训练
```
python distillation/distillation_train.py \
--name tinyclip_yfcc15m_xsmall-dist_ours_base_MoTIS_loss \
--img_aug \
--batch_size 64 \
--dataset_name 'RSTPReid' \
--root_dir './data' \
--num_epoch 60 \
--device "npu:5" \
--distillation \
--temperature 0.02 \
--distillation_MoTIS_temperature 0.5 \
--distillation_inter_ts_kl_temperature 0.5 \
--distillation_inter_ss_kl_temperature 0.5 \
--distillation_inter_ts_sym_kl_temperature 0.5 \
--continue_training "logs/RSTPReid/20250120_215846_sequence_tinyclip_yfcc15m_xsmall-dist_MoTIS-task_itc/best.pth" \
--distillation_config_path "distillation/distillation_xsmall/sequence_ditillation_tinyclip_yfcc15m_xsmall_ours_base_MoTIS_loss.json" \
--teacher_model_config "logs/RSTPReid/20241202_201600_iira/configs.yaml" \
--pretrain_student_choice 'state_dict_pretrain_model/tinyclip_yfcc15m_xsmall.pth' \
>./logs/RSTPReid/training5.log 2>&1 &

```








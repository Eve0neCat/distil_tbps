import argparse


def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')
    parser.add_argument("--device", default="npu:0", help='set training device')

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--continue_training", type=str, default="")
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--pretrain_student_choice", default='state_dict_pretrain_model/ViT-B-16.pt') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=False, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=False, action='store_true', help="whether to use Mask Language Modeling dataset")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='sdm+id+mlm', help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm']")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    
    ######################## distillation ########################
    parser.add_argument("--sequence_distillation_3stages", default=False, action='store_true', help="whether to use sequence distillation") 
    parser.add_argument("--sequence_distillation_2stages", default=False, action='store_true', help="whether to use sequence distillation") 
    parser.add_argument("--distillation", default=False, action='store_true', help="whether to use distillation")     
    parser.add_argument("--distillation_cosin_weight", default=False, action='store_true', help="loss weight to use distillation") 
    parser.add_argument("--distillation_constant_weight", default=False, action='store_true', help="loss weight to use distillation") 
    parser.add_argument("--distillation_weight", type=float, default=0.9, help="distillation loss wieght")

    parser.add_argument("--distillation_config_path", type=str, default="distillation/sequence_distillation_CLIP_6L_MoTIS.json", help="set distillation config path")
    parser.add_argument("--teacher_model_config", type=str, default="logs/CUHK-PEDES/20241202_201442_iira/configs.yaml", help="set teacher config path")
    parser.add_argument("--distillation_pred_temperature", type=float, default=2., help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--itc_loss_alpha", type=float, default=0.3, help="distillation itc loss alpha weight")
    parser.add_argument("--distillation_ssdm_temperature", type=float, default=2., help="initial temperature value")
    parser.add_argument("--distillation_MoTIS_temperature", type=float, default=2., help="initial temperature value")
    parser.add_argument("--distillation_reid_MoTIS_temperature", type=float, default=0.02, help="initial temperature value")
    parser.add_argument("--reid_MoTIS_vision", default=False, action='store_true', help="whether to ditillate vision features")
    parser.add_argument("--reid_MoTIS_text", default=False, action='store_true', help="whether to ditillate text features") 
    parser.add_argument("--distillation_inter_ts_sym_kl_temperature", type=float, default=0.5, help="initial temperature value")
    parser.add_argument("--distillation_inter_ss_kl_temperature", type=float, default=0.5, help="initial temperature value")
    parser.add_argument("--distillation_inter_ts_kl_temperature", type=float, default=0.5, help="initial temperature value")
    parser.add_argument("--distillation_kl_temperature", type=float, default=2., help="initial temperature value")
    parser.add_argument("--distillation_sdm_temperature", type=float, default=2, help="initial temperature value")
    parser.add_argument("--distillation_cross_sdm_temperature", type=float, default=0.02, help="initial temperature value")
    parser.add_argument("--distillation_intra_tch_stu_sdm_temperature", type=float, default=0.5, help="initial temperature value")
    parser.add_argument("--distillation_inter_tch_stu_sdm_temperature", type=float, default=0.02, help="initial temperature value")
    parser.add_argument("--distillation_cross_kl_temperature", type=float, default=0.5, help="initial temperature value")
    parser.add_argument("--distillation_cross_rkl_temperature", type=float, default=0.5, help="initial temperature value")
    parser.add_argument("--distillation_cross_kd_temperature", type=float, default=0.5, help="initial temperature value")
    
    parser.add_argument("--distillation_dsdm_temperature", type=float, default=2., help="initial temperature value")
    parser.add_argument("--dsdm_altha", type=float, default=0.8, help="vision distillation weigth, (1 - dsdm_altha) is text weigth")
    parser.add_argument("--soft_sdm_loss_alpha", type=float, default=0.3, help="soft sdm loss alpha")

    ######################## CLIP-KD ########################
    parser.add_argument('--use_clipkd', action='store_true', help='是否使用CLIP-KD')
    parser.add_argument('--use_fd', action='store_true', help='使用Feature Distillation')
    parser.add_argument('--use_icl', action='store_true', help='使用Interactive Contrastive Learning')
    parser.add_argument('--use_crd', action='store_true', help='使用Contrastive Relational Distillation')
    parser.add_argument('--clipkd_fd_weight', type=float, default=2000.0, help='FD损失权重')
    parser.add_argument('--clipkd_icl_weight', type=float, default=1.0, help='ICL损失权重')
    parser.add_argument('--clipkd_crd_weight', type=float, default=1.0, help='CRD损失权重')
    parser.add_argument('--clipkd_weight', type=float, default=1.0, help='CLIP-KD总权重')
    parser.add_argument('--clipkd_temperature', type=float, default=0.07, help='CLIP-KD温度参数')

    # add sractch model setting
    parser.add_argument("--embed_dim", type=int, default=512)

    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))            
    parser.add_argument("--stride_size", type=int, default=16)
    parser.add_argument("--vision_layers", type=int, default=12)
    parser.add_argument("--vision_width", type=int, default=768)
    parser.add_argument("--vision_patch_size", type=int, default=16)
    parser.add_argument("--vision_heads_size", type=int, default=64)
    parser.add_argument("--vision_from_scratch", default=False, action='store_true', help="whether to train vision encoder from scratch") 

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408) 
    parser.add_argument("--transformer_layers", type=int, default=12)
    parser.add_argument("--transformer_width", type=int, default=512)
    parser.add_argument("--transformer_heads_size", type=int, default=64)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [identity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args()

    return args
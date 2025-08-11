
import sys
sys.path.append('/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS')
import random

import numpy as np
import torch
import argparse
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet.training.network_training.nnUNetTrainer_breast import nnUNetTrainer_breast
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default='3d_fullres') # 网络架构
    parser.add_argument("--network_trainer", default='MAML3_channelTrainerV2BreastRegions') # 网络训练器nnUNetTrainerV2BreastRegions（通道拼接）
    parser.add_argument("--task",default = "Task262_T2DCEReg", help="can be task name or task id") # 任务的 ID
    parser.add_argument("--fold", default='0',help='0, 1, ..., 5 or \'all\'') # 交叉验证的折数，可取值为 0 到 5 或者 all，默认值是 each
    parser.add_argument("-val", "--validation_only", default = True, help="use this if you want to only run the validation",
                        action="store_true") # 只验证 跳过训练阶段
    parser.add_argument("-c", "--continue_training", default = False, help="use this if you want to continue a training",
                        action="store_true") # 若设置该参数，程序会接着之前的训练继续进行
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
                        # default=default_batch_01_plans_identifier, required=False)# 原来
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False) # 若设置该参数，训练数据将不会被解压，读取压缩数据会消耗更多的 CPU 和内存，需谨慎使用。
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true") # 若设置该参数，训练过程将具有确定性，但会显著降低训练速度，不建议使用。

    # 若设置该参数，在验证过程中，程序会导出预测分割结果的 npz 文件，这在进行集成步骤时是必需的
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=True, action="store_true",
                        help="hands off. This is not intended to be used") # 使用最佳的验证模型
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32") # 若设置该参数，将禁用混合精度训练，使用传统的 32 位浮点数进行训练。

    parser.add_argument("--test_folder", required=False, default="validation_raw",
                        help="name of the test folder. No need to use this for most people") # 指定测试文件夹的名称，大多数情况下无需使用此参数。

    # 若设置该参数，程序将不会保存任何参数文件（除了一个临时的检查点，训练结束后会被移除），适用于开发阶段，可节省磁盘空间。
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    # 若设置该参数，将禁用每折训练后的后处理步骤，适用于处理大尺寸图像时。
    parser.add_argument("--disable_postprocessing_on_folds", required=False, default=True, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")

#############################################################################################
    parser.add_argument('--val_disable_overwrite', action='store_false', default=False,
                        help='Validation does not overwrite existing segmentations') # 若设置该参数False，验证过程不会覆盖已有的推理分割结果
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False, # 若设置该参数，将不进行下一阶段的预测
                        help='do not predict next stage')
#############################################################################################

    # 指定预训练模型的检查点文件路径，仅在训练时使用，需谨慎使用。
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')

    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')

    args = parser.parse_args()

    set_seed(args.seed)

    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic
    valbest = args.valbest
    find_lr = args.find_lr
    fp32 = args.fp32

#############################
    run_mixed_precision = not fp32
#############################

    test_folder = args.test_folder

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    elif fold == 'each': # 默认
        pass
    else:
        fold = int(fold) # 第几折

    """
    plans_file: '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/nnUNet_preprocessed/Task082_BraTS2020/nnUNetPlansv2.1_batch_01_plans_3D.pkl'
    output_folder_name: '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task082_BraTS2020/MAML3_channelTrainerV2BraTSRegions__nnUNetPlansv2.1_batch_01'
    dataset_directory: '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/nnUNet_preprocessed/Task082_BraTS2020'
    batch_dice: False
    stage: 0
    trainer_class: <class 'nnunet.training.network_training.MAML3_channelTrainerV2BraTSRegions.MAML3_channelTrainerV2BraTSRegions'>
    """
    plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None: # False
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")



    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data, # False 0 True
                            deterministic=deterministic, # False
                            fp16=run_mixed_precision) # True
    if args.disable_saving: # False
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case the training chashes
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        trainer.save_final_checkpoint = False # whether or not to save the final checkpoint

    trainer.initialize(not validation_only) # False 初始化多模态网络 且 self.was_initialized = True

    if find_lr: # False
        trainer.find_lr()
    else:
        if not validation_only: # True
            if args.continue_training: # False 继续训练
                # -c was set, continue a previous training and ignore pretrained weights
                trainer.load_latest_checkpoint()
            elif (not args.continue_training) and (args.pretrained_weights is not None): # False False 开始新的训练 但是有预训练权重
                # we start a new training. If pretrained_weights are set, use them
                load_pretrained_weights(trainer.network, args.pretrained_weights)
            else:
                # new training without pretraine weights, do nothing 没有预训练权重 重新开始训练
                pass

            trainer.run_training() # that 模型训练
        else:
            if valbest: #默认 True
                trainer.load_best_checkpoint(train=False) # 加载最好的模型
            else:
                trainer.load_final_checkpoint(train=False) # 加载最后保存的模型

        trainer.network.eval()

        trainer.load_best_checkpoint(train=False)
        # predict validation 训练完 模型验证
        trainer.validate(save_softmax=args.npz, validation_folder_name=test_folder,
                         run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                 #################################################
                         overwrite=args.val_disable_overwrite)
                 #################################################



if __name__ == "__main__":
    main()

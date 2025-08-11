
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from torch import nn

from nnunet.evaluation.region_based_evaluation import evaluate_regions, get_breast_regions
from nnunet.training.dataloading.dataset_loading_breast import unpack_dataset
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.dice_loss import DC_and_BCE_loss, get_tp_fp_fn_tn, SoftDiceLoss
from nnunet.training.network_training.MAML3_channelTrainerV2_breast import MAML3_channelTrainerV2_breast
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda

from torch.cuda.amp import autocast


class MAML3_channelTrainerV2BreastRegions_BN(MAML3_channelTrainerV2_breast):
    def initialize_network(self):
        """inference_apply_nonlin to sigmoid"""
        super().initialize_network()
        self.network.inference_apply_nonlin = nn.Sigmoid()


class MAML3_channelTrainerV2BreastRegions(MAML3_channelTrainerV2_breast):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.regions = get_breast_regions() # 肿瘤所在区域（有重叠的）
        self.regions_class_order = (1,) # 标签类别（需要tuple类型）
        self.loss = DC_and_BCE_loss({}, {'batch_dice': False, 'do_bg': True, 'smooth': 0})
    """加载 分割肿瘤类别"""
    def process_plans(self, plans):
        super().process_plans(plans)
        """
        The network has as many outputs as we have regions
        """
        self.num_classes = len(self.regions) # 肿瘤类别

    def initialize_network(self):
        """inference_apply_nonlin to sigmoid"""
        super().initialize_network()
        self.network.inference_apply_nonlin = nn.Sigmoid()
    # 训练初始化权重
    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None): # True
                """
                'pool_op_kernel_sizes': [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],  5个
                'conv_kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]}} 6个
                """
                self.load_plans_file() # 加载预处理后图像pkl -> plans

            self.process_plans(self.plans) # 加载 肿瘤标签 类别

            self.setup_DA_params() # 设置数据增强的相关参数

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes) # 网络中池化操作的数量，也就是深度监督中不同分辨率输出的数量 默认5

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)]) # 随着分辨率的降低，权重呈指数衰减，高分辨率输出的权重更高 [1.  0.5  0.25  0.125  0.0625]

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)]) # 用于屏蔽最低的 1 个输出 [ True  True  True  True False]
            weights[~mask] = 0 # 将 mask 中为 False 的元素对应的权重设置为 0，即屏蔽这些输出 [1.  0.5   0.25  0.125  0.   ]
            weights = weights / weights.sum() # [0.53333333 0.26666667 0.13333333 0.06666667 0.  ]
            # self.ds_loss_weights = weights
            ################################## yao
            self.ds_loss_weights = weights[0] # None -> 0.53333333

            for i in range(self.num_input_channels + 1): # 4+1 0~4 4个图像堆叠  4个通道 + 1背景（不考虑）
                self.ds_loss_weights = np.append(self.ds_loss_weights, weights) # 共 1ds_loss_weights + 25 个weight
            # print('################', self.ds_loss_weights)
            ################################## yao
            
            # now wrap the loss 初始化损失函数
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################
            # ..../nnUNet_preprocessed/Task083_debugBraTS2020/nnUNetData_plans_v2.1_stage0'
            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val, self.dl_ts = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data) # '.../nnUNet_preprocessed/Task083_debugBraTS2020/nnUNetData_plans_v2.1_stage0'
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")
                print(self.data_aug_params) # 打印数据增强参数
                self.tr_gen, self.val_gen, self.ts_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val, self.dl_ts,
                                                                    self.data_aug_params[
                                                                        'patch_size_for_spatialtransform'],
                                                                    self.data_aug_params,
                                                                    deep_supervision_scales=self.deep_supervision_scales,
                                                                    regions=self.regions)  # 进行数据增强
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())), # 写入tr数据 到log  但不打印
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())), # 写入val数据 到log  但不打印
                                       also_print_to_console=False)
                self.print_to_log_file("TEST KEYS:\n %s" % (str(self.dataset_ts.keys())), # 写入ts数据 到log  但不打印
                                    also_print_to_console=False)
            else:
                pass

            self.initialize_network() # 初始化 多模态分割网络
            self.initialize_optimizer_and_scheduler() # 初始化优化器 SGD

        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True # 设置为已经初始化过了
    # 训练完 模型验证过程
    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: int = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                #  segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = False):

        super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)


    """在模型训练过程中对模型的输出进行在线评估，计算前景类别的 Dice 系数（Dice Coefficient），并记录真阳性（TP）、假阳性（FP）和假阴性（FN）的数量"""
    def run_online_evaluation(self, output, target):
        output = output[0] # (B,1,80,192,160)
        target = target[0] # (B,1,80,192,160)
        with torch.no_grad():
            out_sigmoid = torch.sigmoid(output) # (B,1,80,192,160))
            out_sigmoid = (out_sigmoid > 0.5).float()

            if self.threeD: # True
                axes = (0, 2, 3, 4)
            else:
                axes = (0, 2, 3)

            tp, fp, fn, _ = get_tp_fp_fn_tn(out_sigmoid, target, axes=axes)

            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8))) # 前景 dice
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))
    """迭代数据训练"""
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """ tr True False  val False True
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator) # 训练数据 data properties keys target
        data = data_dict['data'] # (B,2,80,192,160)
        target = data_dict['target'] # [ (B,2,80,192,160) (B,2,40,96,80) (B,2,20,48,40) (B,2,10,24,20) (B,2,5,12,10)]

        data = maybe_to_torch(data) # 将输入的数据转换为 torch.Tensor 类型
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)


            ##################################### yao
            if self.epoch % 2:
                tmp_target = [target[0],]

                for i in range(self.num_input_channels + 1):
                    tmp_target += target
                f_target = tmp_target
            else:
                tmp_target = [target[0],] # [(B,1,128,128,128)]

                for i in range(self.num_input_channels + 1):
                    tmp_target += target
                m_target = tmp_target # [(B,1,80,192,160),(B,1,80,192,160),(B,1,40,96,80),(B,1,20,48,40),(B,1,10,24,20),(B,1,5,12,10)]
            #####################################

        self.optimizer.zero_grad()

        if self.fp16: # True
            with autocast():
#################################################################################################
                if self.epoch % 2:
                    f_output = self.network(data)
                    l = self.loss(f_output, f_target)
                else:
                    m_output = self.network(data) # [(B,1,80,192,160) +  3*(B,1,80,192,160),(B,1,40,96,80),(B,1,20,48,40),(B,1,10,24,20),(B,1,5,12,10)]
                    l = self.loss(m_output, m_target) # 计算损失

#################################################################################################



            if do_backprop: # True
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
#################################################################################################
            if self.epoch % 2:
                f_output = self.network(data)
                l = self.loss(f_output, f_target) 
            else:
                m_output= self.network(data)
                l = self.loss(m_output, m_target)
            # l = self.loss(f_output, f_target) + self.loss(m_output, m_target)

#################################################################################################

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
        # 训练时 验证阶段才进入
        if run_online_evaluation:
            if not self.epoch % 2:
                f_output  = self.network(data) # [(1,3,128,128,128) + 5*[(1,3,128,128,128),(1,3,64,64,64),(1,3,32,32,32),(1,3,16,16,16),(1,3,8,8,8)]]
            self.run_online_evaluation(f_output, target)

        del data
        del target

        return l.detach().cpu().numpy()

class MAML3_channelTrainerV2BreastRegions_Dice(MAML3_channelTrainerV2BreastRegions):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = SoftDiceLoss(apply_nonlin=torch.sigmoid, **{'batch_dice': False, 'do_bg': True, 'smooth': 0})

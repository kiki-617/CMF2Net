#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys

import nnunet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.utils import crop
from nnunet.paths import *
import shutil
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.preprocessing.sanity_checks import verify_dataset_integrity
from nnunet.training.model_restore import recursive_find_python_class


def main():
    import argparse

    parser = argparse.ArgumentParser()
    # 任务id
    parser.add_argument("-t", "--task_ids",default=[133,264], nargs="+", help="List of integers belonging to the task ids you wish to run"
                                                            " experiment planning and preprocessing for. Each of these "
                                                            "ids must, have a matching folder 'TaskXXX_' in the raw "
                                                            "data folder")
    parser.add_argument("-pl3d", "--planner3d", type=str, default="ExperimentPlanner3D_v21",
                        help="Name of the ExperimentPlanner class for the full resolution 3D U-Net and U-Net cascade. "
                             "Default is ExperimentPlanner3D_v21. Can be 'None', in which case these U-Nets will not be "
                             "configured")

    parser.add_argument("-no_pp", action="store_true",
                        help="Set this flag if you dont want to run the preprocessing. If this is set then this script "
                             "will only run the experiment planning and create the plans file")
    parser.add_argument("-tl", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the low resolution data for the 3D low "
                             "resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of "
                             "RAM")
    parser.add_argument("-tf", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the full resolution data of the 2D U-Net and "
                             "3D U-Net. Don't overdo it or you will run out of RAM")


    args = parser.parse_args()
    task_ids = args.task_ids
    dont_run_preprocessing = args.no_pp
    tl = args.tl
    tf = args.tf
    planner_name3d = args.planner3d

    if planner_name3d == "None":
        planner_name3d = None

    # we need raw data
    tasks = []

    for i in task_ids:
        i = int(i)

        task_name = convert_id_to_task_name(i) #任务名

        crop(task_name, False, tf) # 裁剪 bbox 图像+标签保存为npz 标签背景值处理为-1

        tasks.append(task_name) # 把当前任务 添加到 任务列表中 （可多任务一起执行）

    search_in = join(nnunet.__path__[0], "experiment_planning")# '..../nnunet/experiment_planning'

    if planner_name3d is not None: # 'ExperimentPlanner3D_v21'
        planner_3d = recursive_find_python_class([search_in], planner_name3d, current_module="nnunet.experiment_planning") # <class 'nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21.ExperimentPlanner3D_v21'>
        if planner_3d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name3d)
    else:
        planner_3d = None

    for t in tasks: # 遍历 任务
        print("\n\n\n", t)
        cropped_out_dir = os.path.join(nnUNet_cropped_data, t) #'..../data/nnUNet_cropped_data/Task083_debugBraTS2020'
        preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t) # '..../data/nnUNet_preprocessed/Task083_debugBraTS2020'

        # we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT
        dataset_json = load_json(join(cropped_out_dir, 'dataset.json')) # 加载'..../data/nnUNet_cropped_data/Task083_debugBraTS2020'下的data.json文件
        modalities = list(dataset_json["modality"].values()) # 模态 ['T1', 'T1ce', 'T2', 'FLAIR']
        collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False # 是否有CT 模态 （当然没有）
        dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=tf)  # 传入 '..../data/nnUNet_cropped_data/Task083_debugBraTS2020'  开始创建数据指纹对象
        _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  #(光保存数据指纹（图像信息）不使用？) this will write output files that will be used by the ExperimentPlanner


        maybe_mkdir_p(preprocessing_output_dir_this_task)
        shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task) # 把数据指纹信息 移动到预处理文件夹 '..../data/nnUNet_preprocessed/Task083_debugBraTS2020'
        shutil.copy(join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task) # 把dataset.json 移动到预处理文件夹 '..../data/nnUNet_preprocessed/Task083_debugBraTS2020'

        threads = (tl, tf)

        print("number of threads: ", threads, "\n")

        if planner_3d is not None:

            exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task) #传入'..../data/nnUNet_cropped_data/Task083_debugBraTS2020' 和 '..../data/nnUNet_preprocessed/Task083_debugBraTS2020' 进入experiment_planner_baseline_3DUNet_v21.py 进行处理
            exp_planner.plan_experiment()
            if not dont_run_preprocessing: # False  # double negative, yooo
                exp_planner.run_preprocessing(threads)



if __name__ == "__main__":
    main()


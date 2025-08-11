import os.path
from typing import Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques

def generate_dataset_json(output_file: str, imagesTr_dir: str,modalities: Tuple,
                          labels: dict, dataset_name: str, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):

    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)
    print(train_identifiers)

    test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [{'image': "./images/%s.nii.gz" %i, "label": "./labels/%s.nii.gz" %i } for i in train_identifiers]
    json_dict['test'] = []

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file))

if __name__ == '__main__':
    target_base = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/nnUNet_raw_data/Task264_CMUexternalValT2DCEReg'
    generate_dataset_json(os.path.join(target_base, 'dataset.json'), os.path.join(target_base, 'images'),
                            ('DCE','T2'),
                          labels={0: 'background', 1: 'tumor'}, dataset_name=os.path.basename(target_base), license='hands off',
                          dataset_description='for CMU DCE  tumor Segmentation.')
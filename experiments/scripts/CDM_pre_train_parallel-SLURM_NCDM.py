from liriscat.CDM import *
from IMPACT import model
from IMPACT.utils import generate_eval_config
from IMPACT.dataset import LoaderDataset as IMPACT_dataset
from liriscat.dataset import preprocessing_utilities as pu
import argparse
from liriscat.utils import convert_config_to_EduCAT
from liriscat.dataset import preprocessing_utilities as pu
from liriscat.CDM.NCDM import NCDM

def main(dataset_name, i_fold=None):

    # Set all the required parameters ---------------
    IMPACT_config = generate_eval_config(num_epochs=200, patience=30, save_params=True, dataset_name=dataset_name,
                                         embs_path="../embs/" + dataset_name, params_path="../ckpt/" + dataset_name,
                                         learning_rate=7.380681029927064e-05, lambda_=2.2656270501845414e-06, batch_size=2048,valid_metric='rmse', pred_metrics=["mi_acc", 'rmse'],profile_metrics=['doa'])

    concept_map, metadata, nb_modalities = pu.load_dataset_resources(IMPACT_config)

    IMPACT_config = convert_config_to_EduCAT(IMPACT_config, metadata)

    IMPACT_config['i_fold'] = i_fold
    vertical_train, vertical_valid = pu.vertical_data(IMPACT_config, i_fold)

    impact_train_data = IMPACT_dataset(vertical_train, concept_map, metadata, nb_modalities)
    impact_valid_data = IMPACT_dataset(vertical_valid, concept_map, metadata, nb_modalities)

    train_data, valid_data = [
        pu.transform(data.raw_data_array[:,0].long(), data.raw_data_array[:,1].long(), concept_map, data.raw_data_array[:,2], IMPACT_config['batch_size'], impact_train_data.n_categories)
        for data in [impact_train_data, impact_valid_data]
    ]

    cdm = NCDM(metadata['num_dimension_id'], metadata['num_item_id'], metadata['num_user_id'], IMPACT_config)
    cdm.train(train_data, valid_data, epoch=IMPACT_config['num_epochs'], device="cuda")
    
    print(cdm.eval(valid_data))

def parse_args():
    parser = argparse.ArgumentParser(description="A program that runs the CDM pre-training session")
    parser.add_argument('dataset_name',help="the dataset name")
    parser.add_argument('--i_fold', type=int, default=None, help="0-indexed fold number (if omitted runs all folds)")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.dataset_name, args.i_fold)

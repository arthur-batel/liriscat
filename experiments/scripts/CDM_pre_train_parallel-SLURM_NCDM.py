from micat.CDM import *
from IMPACT import model
from IMPACT.utils import generate_eval_config
from IMPACT.dataset import LoaderDataset as IMPACT_dataset
from micat.dataset import preprocessing_utilities as pu
import argparse
from micat.CDM.NCDM import CATNCDM

def main(dataset_name, i_fold=None):

    # Set all the required parameters ---------------
    IMPACT_config = generate_eval_config(num_epochs=200, patience=30, save_params=True, dataset_name=dataset_name,
                                         embs_path="../embs/", params_path="../ckpt/",
                                         learning_rate=2.10315236650495e-05, lambda_=2.2656270501845414e-06, batch_size=2048,valid_metric='rmse', pred_metrics=["mi_acc", 'rmse'],profile_metrics=['doa'])

    concept_map, metadata, nb_modalities = pu.load_dataset_resources(IMPACT_config)

    IMPACT_config['i_fold'] = i_fold
    vertical_train, vertical_valid = pu.vertical_data(IMPACT_config, i_fold)

    train_set = IMPACT_dataset(vertical_train, concept_map, metadata, nb_modalities)
    valid_set = IMPACT_dataset(vertical_valid, concept_map, metadata, nb_modalities)

    cdm = CATNCDM(**IMPACT_config)
    cdm.init_CDM_model(train_set,valid_set)
    cdm.train(train_set, valid_set, epoch=IMPACT_config['num_epochs'])
    
    print(cdm.eval(valid_set))

    cdm = CATNCDM(**IMPACT_config)
    cdm.init_CDM_model(train_set,valid_set)
    cdm.train(train_set, valid_set, epoch=IMPACT_config['num_epochs'])
    
    print(cdm.eval(valid_set))

def parse_args():
    parser = argparse.ArgumentParser(description="A program that runs the CDM pre-training session")
    parser.add_argument('dataset_name',help="the dataset name")
    parser.add_argument('--i_fold', type=int, default=None, help="0-indexed fold number (if omitted runs all folds)")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.dataset_name, args.i_fold)

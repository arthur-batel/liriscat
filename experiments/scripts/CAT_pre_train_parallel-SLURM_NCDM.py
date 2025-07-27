from liriscat.CDM import *
from IMPACT import model
from IMPACT.utils import generate_eval_config
from IMPACT.dataset import LoaderDataset as IMPACT_dataset
from liriscat.dataset import preprocessing_utilities as pu
import argparse

def main(dataset_name, i_fold=None):

    # Set all the required parameters ---------------
    IMPACT_config = generate_eval_config(num_epochs=200, save_params=True, dataset_name=dataset_name,
                                         embs_path="../embs/" + dataset_name, params_path="../ckpt/" + dataset_name,
                                         learning_rate=0.0007838402204978467, lambda_=2.2656270501845414e-06, batch_size=2048,valid_metric='mi_acc', pred_metrics=["mi_acc"],profile_metrics=['doa'])

    concept_map, metadata, nb_modalities = pu.load_dataset_resources(IMPACT_config)


    IMPACT_config['i_fold'] = i_fold
    vertical_train, vertical_valid = pu.vertical_data(IMPACT_config, i_fold)

    impact_train_data = IMPACT_dataset(vertical_train, concept_map, metadata, nb_modalities)
    impact_valid_data = IMPACT_dataset(vertical_valid, concept_map, metadata, nb_modalities)

    algo = model.IMPACT(**IMPACT_config)
    algo.init_model(impact_train_data, impact_valid_data)
    algo.train(impact_train_data, impact_valid_data)
    print(algo.evaluate_predictions(impact_valid_data))
    print(algo.evaluate_profiles(impact_valid_data))

def parse_args():
    parser = argparse.ArgumentParser(description="A program that runs the CDM pre-training session")
    parser.add_argument('dataset_name',help="the dataset name")
    parser.add_argument('--i_fold', type=int, default=None, help="0-indexed fold number (if omitted runs all folds)")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.dataset_name, args.i_fold)

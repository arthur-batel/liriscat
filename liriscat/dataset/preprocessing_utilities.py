from collections import defaultdict
from typing import List

import pandas as pd
import random
import numpy as np
import torch

from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

import sys
sys.path.append("../../")

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from IMPACT import utils
utils.set_seed(0)
from IMPACT import dataset
from IMPACT import model
import optuna
import logging
import gc
import json
from importlib import reload

from ipyparallel import Client
import dill

cat_absolute_path = os.path.abspath('../../')

rc = Client()
rc[:].use_dill()
lview = rc.load_balanced_view()


rc[:].execute("import sys; sys.path.append('"+cat_absolute_path+"')")
logging.info("sys.path.append("+cat_absolute_path+")")
with rc[:].sync_imports():
    import json
    from IMPACT import utils, model, dataset
    import logging
    import gc
    import torch

from liriscat import utils as utils_liriscat

def stat_unique(data: pd.DataFrame, key):
    """
    Calculate statistics on unique values in a DataFrame column or combination of columns.

    :param data: A pandas DataFrame.
    :type data: pd.DataFrame
    :param key: The column name or list of column names
    :type key: str or List[str]
    :return: None
    :rtype: None
    """
    if key is None:
        print('Total length: {}'.format(len(data)))
    elif isinstance(key, str):
        print('Number of unique {}: {}'.format(key, len(data[key].unique())))
    elif isinstance(key, list):
        print('Number of unique [{}]: {}'.format(','.join(key), len(data.drop_duplicates(key, keep='first'))))
def remove_duplicates(data: pd.DataFrame, key_attrs: List[str], agg_attrs: List[str]):
    """
    Remove duplicates from a DataFrame based on specified key attributes while aggregating other attributes.

    :param data: Dataset as a pandas.DataFrame
    :type data: pd.DataFrame
def split_data_horizontally(df):
    train = []
    valid = []

    for i_group, group in df.groupby('student_id'):
        group_idxs = group.index.values

        train_item_idx, valid_item_idx = train_test_split(group_idxs, test_size=0.2, shuffle=True)

        train.extend(group.loc[train_item_idx].values.tolist())
        valid.extend(group.loc[valid_item_idx].values.tolist())

    return train, validdef split_data_horizontally(df):
        train = []
        valid = []

        for i_group, group in df.groupby('student_id'):
            group_idxs = group.index.values

            train_item_idx, valid_item_idx = train_test_split(group_idxs, test_size=0.2, shuffle=True)

            train.extend(group.loc[train_item_idx].values.tolist())
            valid.extend(group.loc[valid_item_idx].values.tolist())

        return train, valid    :param key_attrs: Attributes whose combination must be unique
    :type key_attrs: List[str]
    :param agg_attrs: Attributes to aggregate in a set for every unique key
    :type agg_attrs: List[str]
    :return: DataFrame with duplicates removed and specified attributes aggregated
    :rtype: pd.DataFrame
    """
    special_attributes = key_attrs.copy()
    special_attributes.extend(agg_attrs)
    d = {}
    for agg_attr in agg_attrs:
        d.update({agg_attr: set})
    d.update({col: 'first' for col in data.columns if col not in special_attributes})
    data = data.groupby(key_attrs).agg(d).reset_index()
    return data

def split_data_vertically_unique_fold(quadruplet, valid_prop):
    """
    Split data (list of double) into train and validation sets.

    :param quadruplet: List of triplets (sid, qid, score)
    :type quadruplet: list
    :param valid_prop: Fraction of the validation set
    :type valid_prop: float
    :param least_test_length: Minimum number of items a student must have to be included in the test set.
    :type least_test_length: int or None
    :return: Train and validation sets.
    :rtype: list, list, list
    """
    df = pd.DataFrame(quadruplet, columns=["student_id", "item_id", "correct", "dimension_id"])
    df.columns = ["student_id", "item_id", "correct", "dimension_id"]
    df_grouped = df.groupby(['student_id', 'item_id']).agg(
        answers=('correct', list),
        dimensions=('dimension_id', list)
    ).reset_index()

    train_list = []
    valid_list = []

    for i_group, group in df_grouped.groupby('student_id'):
        train_part, valid_part = train_test_split(group, test_size=valid_prop, shuffle=True)
        train_list.append(train_part)
        valid_list.append(valid_part)

    train = pd.concat(train_list, ignore_index=True)
    valid = pd.concat(valid_list, ignore_index=True)

    train_expanded = train.explode(['answers', 'dimensions'])
    valid_expanded = valid.explode(['answers', 'dimensions'])

    return train_expanded, valid_expanded

def are_student_sets_equal(train, valid):
    i_train = train['item_id'].unique()
    i_valid= valid['item_id'].unique()
    return len(set(i_train) - (set(i_valid)))==0

def are_pair_student_equal(train, valid):
    r_train = train.apply(lambda x: str(x['student_id']) + '_' + str(x['item_id']), axis=1).unique()
    r_valid = valid.apply(lambda x: str(x['student_id']) + '_' + str(x['item_id']), axis=1).unique()
    return len(set(r_train) - (set(r_valid))) == 0

def quadruplet_format(data: pd.DataFrame):
    """
    Convert DataFrame into a list of quadruplets with correct data types.

    :param data: Dataset containing columns 'user_id', 'item_id', 'correct', and 'dimension_id'
    :type data: pd.DataFrame
    :return: List of quadruplets [sid, qid, score, dim]
    :rtype: list
    """
    # Ensure data types
    data['user_id'] = data['user_id'].astype(int)
    data['item_id'] = data['item_id'].astype(int)
    data['dimension_id'] = data['dimension_id'].astype(int)
    # 'correct' column might be float or int, depending on your data

    # Extract columns as lists
    user_ids = data['user_id'].tolist()
    item_ids = data['item_id'].tolist()
    corrects = data['correct'].tolist()
    dimension_ids = data['dimension_id'].tolist()

    # Combine columns into a list of quadruplets
    quadruplets = list(zip(user_ids, item_ids, corrects, dimension_ids))

    # Convert each quadruplet to a list
    quadruplets = [list(quad) for quad in quadruplets]

    return quadruplets


def densify(data: pd.DataFrame, grp_by_attr: str, count_attr: str, thd: int):
    """
    Filter out groups in a DataFrame based on a count threshold.

    :param data: Dataset
    :type data: pd.DataFrame
    :param grp_by_attr: Attribute used for grouping
    :type grp_by_attr: str
    :param count_attr: Attribute used for counting within groups
    :type count_attr: str
    :param thd: Threshold for group count (groups with less than thd filtered)
    :type thd: int
    :return: DataFrame with groups filtered out
    """
    n = data.groupby(grp_by_attr)[count_attr].nunique()
    filter = n[n < thd].index.tolist()
    print(f'filter {len(filter)} '+grp_by_attr)

    if len(filter) >0 :
        return data[~data[grp_by_attr].isin(filter)], len(filter)
    else : return data, 0

def create_q2k(data: pd.DataFrame):
    """
    Create mappings from item IDs to sets of dimension IDs and vice versa.

    :param data: Dataset containing 'item_id' and 'dimension_id' columns
    :type data: pd.DataFrame
    :return: item to knowledge mapping and knowledge to item mapping
    :rtype: Dict[str, Set[str]], Dict[str, Set[str]]
    """
    q2k = {}
    table = data.drop_duplicates(subset=["dimension_id","item_id"])
    for i, row in table.iterrows():
        q = int(row['item_id'])
        l = q2k.get(q,[])
        l.append(str(int(row['dimension_id'])))
        q2k[q] = l

    # get knowledge to item map
    k2q = {}
    for q, ks in q2k.items():
        for k in ks:
            k2q.setdefault(k, set())
            k2q[k].add(q)
    return q2k, k2q

def encode_attr(data: pd.DataFrame, attr:str):
    """
    Encode categorical attribute values with numerical IDs.

    :param data: Dataset
    :type data: pd.DataFrame
    :param attr: Attribute to renumber
    :type attr: str
    :return: Encoded DataFrame and mapping from attribute to numerical IDs
    :rtype: pd.DataFrame, Dict[str, int]
    """

    attr2n = {}
    cnt = 0
    for i, row in data.iterrows():
        if row[attr] not in attr2n:
            attr2n[row[attr]] = cnt
            cnt += 1

    data.loc[:, attr] = data.loc[:, attr].apply(lambda x: attr2n[x])
    return data.astype({attr:int}), attr2n


def parse_data(data):
    """
    Parse data into student-based and item-based datasets.

    :param data: List of triplets (sid, qid, score)
    :type data: pd.DataFrame
    :return: Student-based and item-based datasets
    :rtype: defaultdict(dict), defaultdict(dict)
    """

    stu_data = defaultdict(lambda: defaultdict(dict))
    ques_data = defaultdict(lambda: defaultdict(dict))
    for i, row in data.iterrows():
        sid = row.user_id
        qid = row.item_id
        correct = row.correct
        stu_data[sid][qid] = correct
        ques_data[qid][sid] = correct
    return stu_data, ques_data

def quadruplet_format(data: pd.DataFrame):
    """
    Convert DataFrame into a list of quadruplets with correct data types.

    :param data: Dataset containing columns 'user_id', 'item_id', 'correct', and 'dimension_id'
    :type data: pd.DataFrame
    :return: List of quadruplets [sid, qid, score, dim]
    :rtype: list
    """
    # Ensure data types
    data['user_id'] = data['user_id'].astype(int)
    data['item_id'] = data['item_id'].astype(int)
    data['dimension_id'] = data['dimension_id'].astype(int)
    # 'correct' column might be float or int, depending on your data

    # Extract columns as lists
    user_ids = data['user_id'].tolist()
    item_ids = data['item_id'].tolist()
    corrects = data['correct'].tolist()
    dimension_ids = data['dimension_id'].tolist()

    # Combine columns into a list of quadruplets
    quadruplets = list(zip(user_ids, item_ids, corrects, dimension_ids))

    # Convert each quadruplet to a list
    quadruplets = [list(quad) for quad in quadruplets]

    return quadruplets




def one_hot_encoding(df,response_range_dict):
    # Pre-calculate num_copies for each q_name
    df = df[df['item_id'].isin(response_range_dict.keys())]

    df['r_range'] = df['item_id'].map(response_range_dict).astype(int)

    # Initialize an empty list to store the duplicated DataFrames
    dfs = []

    # Vectorized operation to duplicate rows
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        r_range = int(row['r_range'])
        c = int(row['correct'])

        # Create a binary list using numpy
        c_binary_list = np.zeros(r_range, dtype=int)
        c_binary_list[c - 1] = 1

        # Duplicate the row r_range times
        duplicated_rows = pd.DataFrame([row] * r_range)
        duplicated_rows['correct_binary'] = c_binary_list

        # Update item_id with the new values
        duplicated_rows['item_id'] = duplicated_rows['item_id'].astype(
            str) + '_' + duplicated_rows.index.astype(str)

        dfs.append(duplicated_rows)

    # Concatenate all the DataFrames in the list
    return pd.concat(dfs, ignore_index=True)

def rescaling_dict(metadata: pd.DataFrame,q2n):
    """
    Computes response_range_dict, min_response_dict and max_response_dict

    """
    response_range_dict = {}
    min_response_dict = {}
    max_response_dict = {}

    # Iterate over the DataFrame
    for i, row in metadata.iterrows():
        # Extract item ID, min_response, and max_response from the row
        try:
            item_id = q2n[row["Variable Name"]]
            min_response = row["min_response"]
            max_response = row["max_response"]

            # Calculate response range and store it in the dictionary
            response_range = max_response - min_response
            response_range_dict[item_id] = response_range
            min_response_dict[item_id] = min_response
            max_response_dict[item_id] = max_response
        except KeyError as e:
            print(f'{e} were removed from dataset')
    return [response_range_dict, min_response_dict, max_response_dict]

def get_modalities_nb(data, metadata) :

    tensor_data = torch.zeros((metadata['num_user_id'], metadata['num_item_id']), dtype=torch.double)
    sid = torch.from_numpy(data['user_id'].to_numpy()).long()
    qid = torch.from_numpy(data['item_id'].to_numpy()).long()
    val = torch.from_numpy(data['correct'].to_numpy())
    
    tensor_data.index_put_((sid, qid), val)

    R_t = tensor_data
    R_t = R_t.T - 1
    
    nb_modalities = torch.zeros(metadata['num_item_id'], dtype=torch.long)
    
    for item_i, logs in enumerate(R_t):
        unique_logs = torch.unique(logs)
        delta_min = torch.min(
            torch.abs(unique_logs.unsqueeze(0) - unique_logs.unsqueeze(1)) + torch.eye(unique_logs.shape[0]))
        nb_modalities[item_i] = (torch.round(1 / delta_min) + 1).long()
    return nb_modalities

def split_users(df, folds_nb=5, seed=0) :
    """
    k-fold cross validationsplit of users

    """

    users_idx = df['user_id'].unique()
    N = len(users_idx) // 5
    random.Random(seed).shuffle(users_idx)

    train = [[] for _ in range(folds_nb)]
    valid = [[] for _ in range(folds_nb)]
    test = [[] for _ in range(folds_nb)]

    for i_fold in range(folds_nb):
        test_fold, valid_fold = (i_fold - 1) % 5, i_fold

        test_users = users_idx[test_fold * N: (test_fold + 1) * N]
        valid_users = users_idx[valid_fold * N: (valid_fold + 1) * N]
        train_indices = [idx for idx in range(len(users_idx))]
        train_indices = [idx for idx in train_indices if idx //
                         N != test_fold and idx // N != valid_fold]
        train_users = [int(users_idx[idx]) for idx in train_indices]

        train[i_fold] = df[df['user_id'].isin(users_idx[train_users])]
        valid[i_fold] = df[df['user_id'].isin(users_idx[valid_users])]
        test[i_fold] = df[df['user_id'].isin(users_idx[test_users])]



    return train, valid, test

def split_data_horizontally(df):
    train = []
    valid = []

    for i_group, group in df.groupby('user_id'):
        group_idxs = group.index.values

        train_item_idx, valid_item_idx = train_test_split(group_idxs, test_size=0.2, shuffle=True)

        train.extend(group.loc[train_item_idx].values.tolist())
        valid.extend(group.loc[valid_item_idx].values.tolist())

    return train, valid

def save_df_to_csv(data, path):
    """
    Save list of triplets (sid, qid, score) to a CSV file.
    :param data: List of triplets (sid, qid, score)
    :type data: pd.DataFrame
    :param path: Path to CSV file
    :type path: str
    """
    data.to_csv(path, index=False)


def get_metadata(data: pd.DataFrame, keys: List[str]) -> dict:
    m = {}
    for attr in keys:
        m["num_"+attr] = len(data[attr].unique())
    return m

def convert_to_records(data):
    df = data.rename(columns={
        'student_id': 'user_id',
        'answers': 'correct',
        'dimensions': 'dimension_id'
    })
    return df.to_records(index=False, column_dtypes={'user_id': int, 'item_id': int, 'correct': float, 'dimension_id': int})

def load_dataset(config) :

    gc.collect()
    torch.cuda.empty_cache()

    # read datasets
    i_fold = 0
    concept_map = json.load(open(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_concept_map.json', 'r'))
    concept_map = {int(k): [int(x) for x in v] for k, v in concept_map.items()}
    #parameter
    metadata = json.load(open(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_metadata.json', 'r'))
    nb_modalities = torch.load(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_nb_modalities.pkl',weights_only=True)
    train_quadruplets = pd.read_csv(
    f'../datasets/2-preprocessed_data/{config["dataset_name"]}_vert_train_{i_fold}.csv',
    encoding='utf-8', dtype={'student_id': int, 'item_id': int, "correct": float,
                                                             "dimension_id": int})
    valid_quadruplets= pd.read_csv(
    f'../datasets/2-preprocessed_data/{config["dataset_name"]}_vert_valid_{i_fold}.csv',
    encoding='utf-8', dtype={'student_id': int, 'item_id': int, "correct": float,
                                                             "dimension_id": int})

    train_data = dataset.LoaderDataset(convert_to_records(train_quadruplets), concept_map, metadata, nb_modalities)
    valid_data = dataset.LoaderDataset(convert_to_records(valid_quadruplets), concept_map, metadata, nb_modalities)

    return train_data,valid_data,concept_map,metadata



def objective(trial, config, train_data, valid_data):

    gc.collect()
    torch.cuda.empty_cache()

    lr = trial.suggest_float('learning_rate', 1e-5, 5e-2, log=True)
    lambda_param = trial.suggest_float('lambda', 1e-7, 1e-5, log=True)
    #num_responses = trial.suggest_int('num_responses', 9,13)

    config['learning_rate'] = lr
    config['lambda'] = lambda_param
    config['num_responses'] = 12

    algo = model.IMPACT(**config)

    # Init model
    algo.init_model(train_data, valid_data)

    # train model ----
    algo.train(train_data, valid_data)

    best_valid_metric = algo.best_valid_metric

    logging.info("-------Trial number : "+str(trial.number)+"\nBest epoch : "+str(algo.best_epoch)+"\nValues : ["+str(best_valid_metric)+"]\nParams : "+str(trial.params))

    del algo.model
    del algo

    gc.collect()
    torch.cuda.empty_cache()

    return best_valid_metric

def load_dataset_resources(config, base_path: str = "../2-preprocessed_data/"):
    concept_map = json.load(open(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_concept_map.json', 'r'))
    concept_map = {int(k): [int(x) for x in v] for k, v in concept_map.items()}
    #parameter
    metadata = json.load(open(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_metadata.json', 'r'))
    nb_modalities = torch.load(f'../datasets/2-preprocessed_data/{config["dataset_name"]}_nb_modalities.pkl',weights_only=True)
    return concept_map, metadata, nb_modalities

def horizontal_data(config, i_folds):

    train = pd.read_csv(
    f'../datasets/2-preprocessed_data/{config["dataset_name"]}_vert_train_{i_folds}.csv',
    encoding='utf-8', dtype={'student_id': int, 'item_id': int, "correct": float,
                                                             "dimension_id": int})
    valid= pd.read_csv(
    f'../datasets/2-preprocessed_data/{config["dataset_name"]}_vert_valid_{i_folds}.csv',
    encoding='utf-8', dtype={'student_id': int, 'item_id': int, "correct": float,
                                                             "dimension_id": int})

    return convert_to_records(train), convert_to_records(valid)


def load_dataset_new(config) :
    train_data, valid_data, concept_map, metadata = load_dataset(config)
    gc.collect()
    torch.cuda.empty_cache()
    # read datasets
    train_data, valid_data = utils_liriscat.prepare_dataset(config, i_fold=0)

    return train_data,valid_data,concept_map,metadata

def launch_test(trial,train_data,valid_data,config) :

    gc.collect()
    torch.cuda.empty_cache()

    algo = model.IMPACT(**config)

    # Init model
    algo.init_model(train_data, valid_data)

    # train model ----
    algo.train(train_data, valid_data)

    best_valid_metric = algo.best_valid_metric

    logging.info("-------Trial number : "+str(trial.number)+"\nBest epoch : "+str(algo.best_epoch)+"\nValues : ["+str(best_valid_metric)+"]\nParams : "+str(trial.params))

    del algo.model
    del algo

    gc.collect()
    torch.cuda.empty_cache()

    return best_valid_metric


def objective_new(trial, launch_test_var, config, train_data, valid_data):
    rc = Client()
    rc[:].use_dill()
    lview = rc.load_balanced_view()
    lr = trial.suggest_float('learning_rate', 1e-5, 5e-2, log=True)
    lambda_param = trial.suggest_float('lambda', 1e-7, 1e-5, log=True)
    #num_responses = trial.suggest_int('num_responses', 11,13)

    config['learning_rate'] = lr
    config['lambda'] = lambda_param
    config['num_responses'] =12

    return lview.apply_async(launch_test_var,trial,train_data,valid_data, config).get()
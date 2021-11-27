# custom processing for obtaining feature and adjacency matrix
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

def adjacency_matrix(data_path, no_of_users):
    connections = pd.read_csv(data_path)
    adj_matix = [[0 for i in range(no_of_users)] for j in range(no_of_users)]
    for i in tqdm(range(connections.shape[0])):
        adj_matix[connections.iloc[i,0]][connections.iloc[i,1]] = 1
        adj_matix[connections.iloc[i,1]][connections.iloc[i,0]] = 1
    return adj_matix

def feature_matix(features_data_path, targets_path):
    targets = pd.read_csv(targets_path)
    targets = targets.sort_values(['new_id'], ascending=True)
    features = json.load(open(features_data_path, 'r'))
    feature_matrix = []
    for i in range(targets.shape[0]):
        features[str(targets.iloc[i, -1])].extend([targets.iloc[i, 1], targets.iloc[i, 3]])  
        feature_matrix.append(features[str(targets.iloc[i, -1])])
    return feature_matrix, targets

def extract_(language):
    connections_path = f'./{language}/musae_{language}_edges.csv'
    features_path = f'./{language}/musae_{language}_features.json'
    target_path = f'./{language}/musae_{language}_target.csv'
    features, targets = feature_matix(features_path, target_path)
    adj_matrix = adjacency_matrix(connections_path, targets.shape[0])
    return adj_matrix, features, targets

languages = ['ENGB', 'ES', 'FR', 'PTBR', 'RU']
for language in tqdm(languages):
    adj_matrix, featues, targets = extract_(language)
    directories = os.listdir('./data')
    if language not in directories:
        os.mkdir(f'./data/{language}')
    np.save(f'./data/{language}/adjaceny_matrix.npy', adj_matrix)
    np.save(f'./data/{language}/featues.npy', featues)
    np.save(f'./data/{language}/targets.npy', targets)
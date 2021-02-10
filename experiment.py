import os
import pandas as pd
import numpy as np
import time
from glob import glob

# modAl
from modAL.models import ActiveLearner
from modAL.uncertainty import margin_sampling, classifier_entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score
from IPython import display

RANDOM_SEED = 42

def orakle_labeler_kasai(learner, X_pool, X_test, y_test, n_queries=10, k=1, bootstrap=False):
    
    scores = [get_scores(y_test, learner.predict(X_test))]
    query_times = [0] # To get scores and times array at the same length
    teach_time = 0
    query_targets = []
    queries_ids = []
            
    for i in range(n_queries):
        display.clear_output(wait=True) # clear output
        
        print(f'Query {i}, f1-score: {scores[-1][0]}, latency {query_times[-1:]}')
        
        # Store predictions        
        query_indices, query_instances = learner.query(X_pool, k=k)
                       
        # save found indexes
        queries_ids.extend(query_instances['id'])        
        
        # train learner
        instances_match = query_instances.loc[:, ['matching']]
        instances_data = query_instances.drop(['a.index', 'b.index', 'matching', 'id'], axis=1)
        
        if teach_time != 0:
            latency = time.time() - teach_time
            query_times.append(latency)
        teach_time = time.time() # measure time between record is starting to be teached to next item is finished queried for labeling

        learner.teach(instances_data, instances_match.values.ravel(), bootstrap=bootstrap)
        
        # Delete querie instances from pool

        X_pool = X_pool.drop(index=query_instances.index).reset_index(drop=True)

        # save iteration score
        scores.append(get_scores(y_test, learner.predict(X_test)))
    query_times.append(0)
            
    scores_df = pd.DataFrame(scores, columns=['f1', 'precision', 'recall'])
    result = {
        'f1': scores_df['f1'],
        'precision': scores_df['precision'],
        'recall': scores_df['recall'],
        'times': query_times,
        'indexes': np.array(queries_ids).flatten()
    }
    
    return result

# use classifier to select instances to label. Keep track of id (original index)
def kasai_query(classifier, X_pool, k=1):

    X_features = X_pool.drop(columns=['a.index', 'b.index', 'matching', 'id'])
    
    predict_proba = classifier.predict_proba(X_features)

    preds = pd.DataFrame({
        'index': X_pool.index,                      # pool index
        'prediction': classifier.predict(X_features),      # 0/1 prediction
        'score0': predict_proba[:,0],                  # probability (0-1) that it is non-match
        'score1': predict_proba[:,1],                  # probability that it is a match
        'entr': classifier_entropy(classifier, X_features) # entropy (no currently used)
    })

    # split into predicted match and non matches. Sort by entropy?
    pred0 = preds.loc[preds['score0'] > 0.5, :].sort_values(by=['score0'], ascending=False)
    pred1 = preds.loc[preds['score1'] >= 0.5, :].sort_values(by=['score1'], ascending=False) 

    # Take top k and bottom k from each subset.

    # high confidence (high score, low entropy)
    hc0 = pred0.iloc[:k].loc[:, 'index']
    hc1 = pred1.iloc[:k].loc[:, 'index']

    # uncertain (low score - close to 0.5, high entropy)
    un0 = pred0.iloc[-k:].loc[:, 'index']
    un1 = pred1.iloc[-k:].loc[:, 'index']

    query_instances = pd.concat([hc0, hc1, un0, un1])
        
    return query_instances


def do_al_kasai(dataset, data, queries, n_initial=8, k=1):
    np.random.seed(RANDOM_SEED)
    print(f'Dataset: {dataset}')
    
    learner, initial_idx, X_pool, X_test, y_test = initialize_modal_learner(data['train'], data['test'],
                                                                                               query_strategy=kasai_query,
                                                                                               n_initial=n_initial)
    result = orakle_labeler_kasai(learner, X_pool, X_test, y_test, n_queries=queries, k=k)
           
        
    result['initial_indexes'] = initial_idx
    
    return result

# Used to adjust number of iterations when using k (4 queries for each iteration)
def get_kasai_queries(num_samples, n_initial=8, k=1):
    q = num_samples - n_initial
    return q // 4*k

def get_scores(y_true, y_pred):
    return [scorer(y_true, y_pred) for scorer in [f1_score, precision_score, recall_score]]

def initialize_modal_learner(X_train, X_test, query_strategy, n_initial=8):
    y_train = X_train.loc[:, ['matching']]

    # We use sss to be sure we have both classes in the training data
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_initial)#, random_state=RANDOM_SEED)
    initial_idx, _ = next(sss.split(X_train, y_train))
    
    # Fix for when sss would only give one class (unbalanced set); sample randomly
    while y_train.iloc[initial_idx].sum().sum() == 0:
        initial_idx = np.random.choice(range(len(y_train)), size=n_initial)

    learner = ActiveLearner(
        estimator=RandomForestClassifier(n_jobs=-1, random_state=RANDOM_SEED),
        query_strategy=query_strategy,
        X_training=X_train.iloc[initial_idx].drop(['a.index', 'b.index', 'matching', 'id'], axis=1), 
        y_training=y_train.iloc[initial_idx].values.ravel(),
    )
    
    X_pool = X_train.drop(X_train.index[initial_idx]).reset_index(drop=True)

    y_test = X_test['matching']
    X_test = X_test.drop(['a.index', 'b.index', 'matching', 'id'], axis=1) 
    
    return learner, initial_idx, X_pool, X_test, y_test




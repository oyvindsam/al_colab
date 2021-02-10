import os, sys
import pandas as pd

from al_colab.dataset import *
from al_colab.pyem_utils import *
from al_colab.experiment import *

def write_results(result, dataset, path, filename):
    out_dir_idx = os.path.join('out', path)
    out_file = os.path.join(out_dir_idx, filename)

    if not os.path.exists(out_dir_idx):
        try:
            os.makedirs(out_dir_idx)
        except OSError as e:
            print(e.strerror)
    
    indexes =  pd.DataFrame({
        dataset: np.concatenate((result['initial_indexes'], result['indexes'])),
    })
        
    data = pd.DataFrame({
        'f1': result['f1'],
        'precision': result['precision'],
        'recall': result['recall'],
        'times': result['times'],
    })
    
    indexes.to_csv(out_file, index=False)
    data.to_csv(out_file+'_data', index=True)
    
    
def run_al_all(n_samples=None, all=False, n_initial=8, k=1):
    n_queries = 0
    
    ds = load_datasets(all=all)
    # normal AL
    for dataset in ds:
        print(f'Dataset: {dataset}.')
        
        d = ds[dataset]
        d.load()
        
        X_train, X_test = create_train_test(d, additional_features=None)
        
        data = {
            'train' : X_train,
            'test' : X_test,
        }
        
        n = n_samples if n_samples is not None else len(d.matches_train)
        n_queries = get_kasai_queries(n)
            
        result = do_al_kasai(dataset, data, n_queries)

        # write to file
        filename = f"{n_initial}-{n}-{k}" # initial - queries - k

        write_results(result, dataset, d.name, filename)

# load all datasets in a dictionary
def load_datasets(all=False):
    data = {}
    datasets = dataset_all if all else dataset_small
    
    for key in datasets:
        data[key] = (datasets[key]())
    
    return data  

dataset_all = {
    #'deepmatcher_structured_amazon_google': deepmatcher_structured_amazon_google,
    #'deepmatcher_structured_beer':deepmatcher_structured_beer,  #fails
    #'deepmatcher_structured_dblp_acm': deepmatcher_structured_dblp_acm,
    #'deepmatcher_structured_dblp_google_scholar': deepmatcher_structured_dblp_google_scholar,
    #'deepmatcher_structured_fodors_zagats': deepmatcher_structured_fodors_zagats,
    'deepmatcher_structured_walmart_amazon': deepmatcher_structured_walmart_amazon,  #fails
    #'deepmatcher_structured_itunes_amazon': deepmatcher_structured_itunes_amazon,
    #'deepmatcher_dirty_dblp_acm': deepmatcher_dirty_dblp_acm,
    #'deepmatcher_dirty_dblp_google_scholar': deepmatcher_dirty_dblp_google_scholar, 
    #'deepmatcher_dirty_walmart_amazon':deepmatcher_dirty_walmart_amazon,   fails
    #'deepmatcher_dirty_itunes_amazon':deepmatcher_dirty_itunes_amazon,
    #'deepmatcher_textual_abt_buy':deepmatcher_textual_abt_buy,
    #'deepmatcher_textual_company':deepmatcher_textual_company,  uses too much time
    #'comperbench_abt_buy':comperbench_abt_buy,
    #'comperbench_wdc_xlarge_shoes':comperbench_wdc_xlarge_shoes,   fails
}

dataset_small = {
    #'deepmatcher_textual_abt_buy':deepmatcher_textual_abt_buy,
    #'deepmatcher_structured_itunes_amazon': deepmatcher_structured_itunes_amazon,
    #'deepmatcher_structured_amazon_google': deepmatcher_structured_amazon_google,
    #'deepmatcher_structured_dblp_acm': deepmatcher_structured_dblp_acm,
    'deepmatcher_structured_dblp_google_scholar': deepmatcher_structured_dblp_google_scholar,
}
import py_entitymatching as em
import os
import pandas as pd
import pyarrow as pa
import numpy as np
import dask
import warnings
import time
from glob import glob

def create_train_test(d, main_column='feature1', additional_features=['feature2']):  
         
    # check if all additional features is in at least 1 column
    include_additional = additional_features is not None and \
        all(any([f == af for f in d.records_a.columns.values]) for af in additional_features)
    
    columns = [main_column]
    if include_additional:
        columns.extend(additional_features)

    X_train = create_feature_vectors(d.matches_train, d.records_a, d.records_b, columns)
    X_test = create_feature_vectors(d.matches_test, d.records_a, d.records_b, columns)
    
    return X_train, X_test


def create_feature_vectors(C, l, r, columns):

    l = l.loc[:, columns]
    r = r.loc[:, columns]
    
    C['id'] = C.index
    l['a.index'] = l.index
    r['b.index'] = r.index
    
    setup_keys(C, l, r)

    atypes_l = em.get_attr_types(l)
    atypes_r = em.get_attr_types(r)
    
    for c in columns:
        if atypes_l[c] != atypes_r[c]: # how to do this more gracefully? 
            atypes_r[c] = 'str_bt_5w_10w'
            atypes_l[c] = 'str_bt_5w_10w'

    corres = em.get_attr_corres(l, r)

    tok = em.get_tokenizers_for_blocking()
    sim = em.get_sim_funs_for_blocking()
    
    feature_table = em.get_features(l, r, atypes_l, atypes_r, corres, tok, sim)
    
    # Generate features
    X = get_feature_vectors(C, feature_table, attrs_before=['matching'])

    return X

def get_feature_vectors(C, feature_table, attrs_before=None, attrs_after=None):
    
    H = em.extract_feature_vecs(C, 
                                feature_table=feature_table, 
                                attrs_before=attrs_before,
                                attrs_after=attrs_after,
                                show_progress=True,
                                n_jobs=-1)
    # Set NaNs to 0
    H.fillna(0, inplace=True)
    
    return H

def setup_keys(C, l, r):
    em.set_key(C, 'id')
    em.set_key(l, 'a.index')
    em.set_key(r, 'b.index')
    em.set_fk_ltable(C, 'a.index')
    em.set_fk_rtable(C, 'b.index')
    em.set_ltable(C, l)
    em.set_rtable(C, r)
    
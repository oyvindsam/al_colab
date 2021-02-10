# Dere kan fikse datasett slik dere selv ønsker det, men her litt kode for å laste ned noen vanlige datasett

import os
import urllib.request
import logging
from functools import lru_cache
import zipfile
from glob import glob
import pandas as pd

log = logging.getLogger(__name__)


class EntityMatchingDataset:
    def __init__(self, name, data_dir = None):
        self.name = name
        if not data_dir:
            data_dir = os.path.join(os.getcwd(), "data")
        self.data_dir = data_dir
    
    @property
    @lru_cache(maxsize=1)
    def records_a(self):
        self.download()
        return self._load("records_a")
    
    @property
    @lru_cache(maxsize=1)
    def records_b(self):
        self.download()
        return self._load("records_b")
    
    @property
    @lru_cache(maxsize=1)
    def matches_train(self):
        self.download()
        return self._load("matches_train")
    
    @property
    @lru_cache(maxsize=1)
    def matches_val(self):
        self.download()
        return self._load("matches_val")
    
    @property
    @lru_cache(maxsize=1)
    def matches_test(self):
        self.download()
        return self._load("matches_test")
    
    def _download_file(self, url, filename):
        file_path = os.path.join(self.data_dir, self.name, filename)
        if not os.path.exists(file_path):
            log.warning(f"Downloading {url} to {file_path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            urllib.request.urlretrieve(url, file_path)
    
    def download(self):
        ...
        
    def load(self):
        self.records_a
        self.records_b
        self.matches_train
        self.matches_val
        self.matches_test
        
class DeepMatcherDataset(EntityMatchingDataset):
    def __init__(self, name, url, data_dir = None):
        super().__init__(name, data_dir)
        self._url = url
    
    def download(self):
        self._download_file(urllib.parse.urljoin(self._url, "tableA.csv"), "tableA.csv")
        self._download_file(urllib.parse.urljoin(self._url, "tableB.csv"), "tableB.csv")
        self._download_file(urllib.parse.urljoin(self._url, "train.csv"), "train.csv")
        self._download_file(urllib.parse.urljoin(self._url, "valid.csv"), "valid.csv")
        self._download_file(urllib.parse.urljoin(self._url, "test.csv"), "test.csv")
    
    def _load(self, t):
        names = {
            'title': 'feature1',
            'Beer_Name': 'feature1',
            'Song_Name': 'feature1',
            'content': 'feature1',
            'manufacturer': 'feature2',
            'Brew_Factory_Name': 'feature2',
            'authors': 'feature2',
            'addr': 'feature2',  
            'Artist_Name': 'feature2',
            'modelno': 'feature2',    
        }
        
        filename = {
            "records_a": "tableA.csv",
            "records_b": "tableB.csv",
            "matches_train": "train.csv",
            "matches_val": "valid.csv",
            "matches_test": "test.csv",
        }[t]
        if t.startswith("records"):
            d = pd.read_csv(os.path.join(self.data_dir, self.name, filename), index_col="id").rename_axis(index="index")
            d = d.rename(columns=lambda s: s if s not in names else names[s])
            return d
        else:
            d = pd.read_csv(os.path.join(self.data_dir, self.name, filename)).rename(columns={"ltable_id": "a.index", "rtable_id": "b.index", "label": "matching"}).astype({"matching": "bool"})
            d = d.rename(columns=lambda s: s if s not in names else names[s])
            return d

def deepmatcher_structured_amazon_google(data_dir=None):
    return DeepMatcherDataset("DeepMatcher/Structured/Amazon-Google", "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/exp_data/", data_dir)
        
def deepmatcher_structured_beer(data_dir=None):
    return DeepMatcherDataset("DeepMatcher/Structured/Beer", "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Beer/exp_data/", data_dir)

def deepmatcher_structured_dblp_acm(data_dir=None):
    return DeepMatcherDataset("DeepMatcher/Structured/DBLP-ACM", "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-ACM/exp_data/", data_dir)

def deepmatcher_structured_dblp_google_scholar(data_dir=None):
    return DeepMatcherDataset("DeepMatcher/Structured/DBLP-GoogleScholar", "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-GoogleScholar/exp_data/", data_dir)
    
def deepmatcher_structured_fodors_zagats(data_dir=None):
    return DeepMatcherDataset("DeepMatcher/Structured/Fodors-Zagats", "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Fodors-Zagats/exp_data/", data_dir)
    
def deepmatcher_structured_walmart_amazon(data_dir=None):
    return DeepMatcherDataset("DeepMatcher/Structured/Walmart-Amazon", "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/exp_data/", data_dir)
    
def deepmatcher_structured_itunes_amazon(data_dir=None):
    return DeepMatcherDataset("DeepMatcher/Structured/iTunes-Amazon", "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/iTunes-Amazon/exp_data/", data_dir)

    
def deepmatcher_dirty_dblp_acm(data_dir=None):
    return DeepMatcherDataset("DeepMatcher/Dirty/DBLP-ACM" ,"http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/DBLP-ACM/exp_data/", data_dir)

def deepmatcher_dirty_dblp_google_scholar(data_dir=None):
    return DeepMatcherDataset("DeepMatcher/Dirty/DBLP-GoogleScholar", "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/DBLP-GoogleScholar/exp_data/", data_dir)
    
def deepmatcher_dirty_walmart_amazon(data_dir=None):
    return DeepMatcherDataset("DeepMatcher/Dirty/Walmart-Amazon", "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/Walmart-Amazon/exp_data/", data_dir)
    
def deepmatcher_dirty_itunes_amazon(data_dir=None):
    return DeepMatcherDataset("DeepMatcher/Dirty/iTunes-Amazon", "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/iTunes-Amazon/exp_data/", data_dir)

    
def deepmatcher_textual_abt_buy(data_dir=None):
    return DeepMatcherDataset("DeepMatcher/Textual/Abt-Buy", "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/exp_data/", data_dir)
    
def deepmatcher_textual_company(data_dir=None):
    return DeepMatcherDataset("DeepMatcher/Textual/Company", "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Company/exp_data/", data_dir)


class CompERBenchDataset(EntityMatchingDataset):
    def __init__(self, name, url, deduplication=False, data_dir = None):
        super().__init__(name, data_dir)
        self._url = url
        self._deduplication = deduplication
    
    def download(self):
        self._download_file(urllib.parse.urljoin(self._url, "records.zip"), "records.zip")
        if not os.path.exists(os.path.join(self.data_dir, self.name, "record_descriptions")):
            with zipfile.ZipFile(os.path.join(self.data_dir, self.name, "records.zip")) as zip_ref:
                zip_ref.extractall(os.path.join(self.data_dir, self.name))
        self._download_file(urllib.parse.urljoin(self._url, "gs_train.csv"), "gs_train.csv")
        self._download_file(urllib.parse.urljoin(self._url, "gs_val.csv"), "gs_val.csv")
        self._download_file(urllib.parse.urljoin(self._url, "gs_test.csv"), "gs_test.csv")
    
    def _load(self, t):
        names = {
            'title': 'name',
            'Beer_Name': 'name',
            'Song_Name': 'name',
            'content': 'name',
            'Brew_Factory_Name': 'description',
            'authors':'description',
            'addr':'description',  
            'Artist_Name':'description',
            'modelno':'description',    
        }
        
        path = {
            "records_a": glob(os.path.join(self.data_dir, self.name, "record_descriptions", "1_*"))[0],
            "records_b": glob(os.path.join(self.data_dir, self.name, "record_descriptions", "2_*" if not self._deduplication else "1_*"))[0],
            "matches_train": os.path.join(self.data_dir, self.name, "gs_train.csv"),
            "matches_val": os.path.join(self.data_dir, self.name, "gs_val.csv"),
            "matches_test": os.path.join(self.data_dir, self.name, "gs_test.csv"),
        }[t]
        if t.startswith("records"):
            d = pd.read_csv(path, index_col="subject_id", encoding="iso-8859-1").rename_axis(index="index")
            d = d.rename(columns=lambda s: s if s not in names else names[s])
            return d
        else:
            d = pd.read_csv(path, encoding="iso-8859-1").rename(columns={"source_id": "a.index", "target_id": "b.index"}).astype({"matching": bool})
            d = d.rename(columns=lambda s: s if s not in names else names[s])
            return d
    
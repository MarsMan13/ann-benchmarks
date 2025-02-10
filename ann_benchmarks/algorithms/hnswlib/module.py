import hnswlib
import numpy as np

from ..base.module import BaseANN


class HnswLib(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        # print(self.method_param,save_index,query_param)
        # self.ef=query_param['ef']
        self.name = "hnswlib (%s)" % (self.method_param)

    def fit(self, X):
        # Only l2 is supported currently
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        # CGCG : validate this
        # self.p.set_num_threads(8)
        self.p.init_index(
            max_elements=len(X), ef_construction=self.method_param["efConstruction"], M=self.method_param["M"]
        )
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        # self.p.set_num_threads(1)

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)

    def query(self, v, n):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        # return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]
        return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0]
    
    def batch_query(self, X, n):
        # CGCG : validate this
        # self.p.set_num_threads(8)
        self.res = self.p.knn_query(X, k=n)
        
    def get_batch_results(self):
        I, D = self.res
        return I

    def freeIndex(self):
        del self.p

import hnswlib
import numpy as np

from ..base.module import BaseANN


class HnswLib(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        self.name = "hnswlib (%s)" % (self.method_param)

    def fit(self, X):
        # Only l2 is supported currently
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        # TODO,CGCG : Knob
        # self.p.set_num_threads(16)
        self.p.init_index(
            max_elements=len(X), ef_construction=self.method_param["efConstruction"], M=self.method_param["M"]
        )
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        # TODO,CGCG : Knob
        self.p.set_num_threads(1)

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)

    def query(self, v, n):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        # return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]
        return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0]
    
    # Batch Logic 1 
    # def batch_query(self, X, n):
    #     # TODO,CGCG : Knob
    #     # self.p.set_num_threads(8)
    #     self.res = self.p.knn_query(X, k=n)
    
    # def get_batch_results(self):
    #     I, D = self.res
    #     return I
    
    # Batch Logic 2
    def get_batch_results(self):
        I = np.squeeze(np.array(self.res), axis=1)
        return I

    def freeIndex(self):
        del self.p

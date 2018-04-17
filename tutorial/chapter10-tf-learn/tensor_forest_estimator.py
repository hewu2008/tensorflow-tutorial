#encoding:utf8
import tensorflow.contrib.learn as estimator

class TensorForestEstimator(estimator.BaseEstimator):
    def __init__(self, params, device_assigner=None, model_dir=None, graph_builder_class = tensor_forest.RandomForestGraphs, master='',accuracy_metric=None, tf_random_seed=None, config=None):
        self.param = params.fill()
        self.accuracy_metric = (accuracy_metric)h
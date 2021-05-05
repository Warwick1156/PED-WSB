import warnings
from lime.lime_tabular import LimeTabularExplainer
from lime import submodular_pick

# https://lime-ml.readthedocs.io/en/latest/lime.html

class LIME:
    def __init__(self, X, predict_fn, num_features=5, features_names=None, result_label='score', categorical_features=None):

        self.explainer = LimeTabularExplainer(X, 
            feature_names=features_names,
            class_names=[result_label],
            mode='regression',
            categorical_features=categorical_features)

        self.predict_fn = predict_fn
        self.num_features = num_features
        self.splime = None

    
    def explain_instance(self, x):
        return self.explainer.explain_instance(x, self.predict_fn, num_features=self.num_features)


    def fit(self, X, sample_size=20, num_expected_examples=15):
        # https://github.com/marcotcr/lime/blob/master/doc/notebooks/Submodular%20Pick%20examples.ipynb
        self.splime = submodular_pick.SubmodularPick(self.explainer, X, 
            self.predict_fn, sample_size=sample_size, 
            num_features=self.num_features, num_exps_desired=num_expected_examples)

    
    def get_explanations(self):
        return self.splime.sp_explanations
import context
import numpy as np
from catboost import CatBoostClassifier


class ClassificationEnsemble(object):

    def __init__(self, esize=10, iterations=1000, lr=0.1, random_strength=None, border_count=None, depth=6, seed=100,
                 load_path=None, task_type=None, devices=None, verbose=True, use_base_model=False,
                 max_ctr_complexity=None, posterior_sampling=True):

        self.seed = seed
        self.esize = esize
        self.depth = depth
        self.iterations = iterations
        self.lr = lr
        self.posterior_sampling = posterior_sampling
        if self.posterior_sampling:
            self.bootstrap_type = 'No'
            self.subsample = None
        else:
            self.bootstrap_type = 'Bernoulli'
            self.subsample = 0.5
        self.random_strength = random_strength
        self.border_count = border_count
        self.posterior_sampling = posterior_sampling
        self.ensemble = []
        self.use_best_model = use_base_model
        for e in range(self.esize):
            model = CatBoostClassifier(iterations=self.iterations,
                                       depth=self.depth,
                                       learning_rate=self.lr,
                                       border_count=self.border_count,
                                       random_strength=self.random_strength,
                                       loss_function='MultiClass',
                                       verbose=verbose,
                                       posterior_sampling=self.posterior_sampling,
                                       use_best_model=self.use_best_model,
                                       max_ctr_complexity=max_ctr_complexity,
                                       random_seed=self.seed + e,
                                       subsample=self.subsample,
                                       task_type=task_type,
                                       bootstrap_type=self.bootstrap_type,
                                       devices=devices)
            self.ensemble.append(model)

        if load_path is not None:
            for i, m in enumerate(self.ensemble):
                m.load_model(f"{load_path}/model{i}.cbm")

    def fit(self, data, eval_set=None, save_path="./"):

        for i, m in enumerate(self.ensemble):
            print(f"TRAINING MODEL {i}\n\n")
            m.fit(data, eval_set=eval_set)
            m.save_model(f"{save_path}/model{i}.cbm")
            assert np.all(m.classes_ == self.ensemble[0].classes_)

    def save(self, path):
        for i, m in enumerate(self.ensemble):
            m.save_model(f"{path}/model{i}.cmb")

    def predict(self, x):
        probs = []

        for m in self.ensemble:
            assert np.all(m.classes_ == self.ensemble[0].classes_)
            prob = m.predict_proba(x)
            probs.append(prob)

        probs = np.stack(probs)
        return probs


class ClassificationEnsembleSGLB(ClassificationEnsemble):

    def __init__(self, esize=10, iterations=1000, lr=0.1, random_strength=None, border_count=None, depth=6, seed=100,
                 load_path=None, task_type=None, devices=None, verbose=True, use_base_model=False,
                 max_ctr_complexity=None, n_objects=10000):

        self.seed = seed
        self.esize = esize
        self.depth = depth
        self.iterations = iterations
        self.lr = lr
        self.random_strength = random_strength
        self.border_count = border_count
        self.ensemble = []
        self.use_best_model = use_base_model
        self.n_objects = n_objects
        for e in range(self.esize):
            model = CatBoostClassifier(iterations=self.iterations,
                                       depth=self.depth,
                                       learning_rate=self.lr,
                                       border_count=self.border_count,
                                       random_strength=self.random_strength,
                                       loss_function='MultiClass',
                                       verbose=verbose,
                                       langevin=True,
                                       diffusion_temperature=self.n_objects,
                                       model_shrink_rate=0.5/self.n_objects,
                                       use_best_model=self.use_best_model,
                                       max_ctr_complexity=max_ctr_complexity,
                                       random_seed=self.seed + e,
                                       subsample=None,
                                       task_type=task_type,
                                       bootstrap_type='No',
                                       devices=devices)
            self.ensemble.append(model)

        if load_path is not None:
            for i, m in enumerate(self.ensemble):
                m.load_model(f"{load_path}/model{i}.cbm")

    def fit(self, data, eval_set=None, save_path="./"):

        for i, m in enumerate(self.ensemble):
            print(f"TRAINING MODEL {i}\n\n")
            m.fit(data, eval_set=eval_set)
            m.save_model(f"{save_path}/model{i}.cbm")
            assert np.all(m.classes_ == self.ensemble[0].classes_)

    def save(self, path):
        for i, m in enumerate(self.ensemble):
            m.save_model(f"{path}/model{i}.cmb")

    def predict(self, x):
        probs = []

        for m in self.ensemble:
            assert np.all(m.classes_ == self.ensemble[0].classes_)
            prob = m.predict_proba(x)
            probs.append(prob)

        probs = np.stack(probs)
        return probs
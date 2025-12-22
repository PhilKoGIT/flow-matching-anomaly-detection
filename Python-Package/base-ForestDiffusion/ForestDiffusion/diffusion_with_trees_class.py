import math
import numpy as np
from ForestDiffusion.utils.diffusion import VPSDE, get_pc_sampler
import importlib
import ForestDiffusion
#importlib.reload(ForestDiffusion)
#from ForestDiffusion import ForestDiffusionModel
import copy
import xgboost as xgb
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from ForestDiffusion.utils.utils_diffusion import build_data_xt, euler_solve, IterForDMatrix, euler_solve_from_x_t, get_xt
from joblib import delayed, Parallel
from scipy.special import softmax

## Class for the flow-matching or diffusion model
# Categorical features should be numerical (rather than strings), make sure to use x = pd.factorize(x)[0] to make them as such
# Make sure to specific which features are categorical and which are integers
# Note: Binary features can be considered integers since they will be rounded to the nearest integer and then clipped
class ForestDiffusionModel():
  def __init__(self, 
               X, # Numpy dataset 
               X_covs=None, # Numpy dataset of additional covariates/features in order to sample X | X_covs (Optional); note that these variables will not be transformed, please apply your own z-scoring or min-max scaling if desired.
               label_y=None, # must be a categorical/binary variable; if provided will learn multiple models for each label y
               n_t=50, # number of noise level
               model='xgboost', # xgboost, random_forest, lgbm, catboost
               diffusion_type='flow', # vp, flow (flow is better, but only vp can be used for imputation)
               max_depth = 7, n_estimators = 100, eta=0.3, # xgboost hyperparameters
               tree_method='hist', reg_alpha=0.0, reg_lambda = 0.0, subsample=1.0, # xgboost hyperparameters
               num_leaves=31, # lgbm hyperparameters
               duplicate_K=100, # number of different noise sample per real data sample
               bin_indexes=[], # vector which indicates which column is binary
               cat_indexes=[], # vector which indicates which column is categorical (>=3 categories)
               int_indexes=[], # vector which indicates which column is an integer (ordinal variables such as number of cats in a box)
               remove_miss=False, # If True, we remove the missing values, this allow us to train the XGBoost using one model for all predictors; otherwise we cannot do it
               p_in_one=True, # When possible (when there are no missing values), will train the XGBoost using one model for all predictors
               true_min_max_values=None, # Vector of form [[min_x, min_y], [max_x, max_y]]; If  provided, we use these values as the min/max for each variables when using clipping
               gpu_hist=False, # using GPU or not with xgboost
               n_z=10, # number of noise to use in zero-shot classification
               eps=1e-3, 
               beta_min=0.1, 
               beta_max=8, 
               n_jobs=-1, # cpus used (feel free to limit it to something small, this will leave more cpus per model; for lgbm you have to use n_jobs=1, otherwise it will never finish)
               n_batch=1, # If >0 use the data iterator with the specified number of batches
               seed=666,
               **xgboost_kwargs): # you can pass extra parameter for xgboost

    assert isinstance(X, np.ndarray), "Input dataset must be a Numpy array"
    assert len(X.shape)==2, "Input dataset must have two dimensions [n,p]"
    assert diffusion_type == 'vp' or diffusion_type == 'flow'
    if X_covs is not None:
      assert X_covs.shape[0] == X.shape[0]
    np.random.seed(seed)

    # Sanity check, must remove observations with only missing data
    obs_to_remove = np.isnan(X).all(axis=1)
    X = X[~obs_to_remove]
    if label_y is not None:
      label_y = label_y[~obs_to_remove]

    # Remove all missing values
    obs_to_remove = np.isnan(X).any(axis=1)
    if remove_miss or (obs_to_remove.sum() == 0):
      X = X[~obs_to_remove]
      if label_y is not None:
        label_y = label_y[~obs_to_remove]
      self.p_in_one = p_in_one # All variables p can be predicted simultaneously
    else:
      self.p_in_one = False

    int_indexes = int_indexes + bin_indexes # since we round those, we do not need to dummy-code the binary variables

    if true_min_max_values is not None:
        self.X_min = true_min_max_values[0]
        self.X_max = true_min_max_values[1]
    else:
        self.X_min = np.nanmin(X, axis=0, keepdims=1)
        self.X_max = np.nanmax(X, axis=0, keepdims=1)

    self.cat_indexes = cat_indexes
    self.int_indexes = int_indexes
    if len(self.cat_indexes) > 0:
        X, self.X_names_before, self.X_names_after = self.dummify(X) # dummy-coding for categorical variables

    # min-max normalization, this applies to dummy-coding too to ensure that they become -1 or +1
    self.scaler = MinMaxScaler(feature_range=(-1, 1))
    X = self.scaler.fit_transform(X)

    X1 = X
    self.X_covs = X_covs
    self.X1 = copy.deepcopy(X1)
    self.b, self.c = X1.shape
    if X_covs is not None:
      self.c_all = X1.shape[1] + X_covs.shape[1]
    else:
      self.c_all = X1.shape[1]
    self.n_t = n_t
    self.duplicate_K = duplicate_K
    self.model = model
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.seed = seed
    self.num_leaves = num_leaves
    self.eta = eta
    self.gpu_hist = gpu_hist
    self.label_y = label_y
    self.n_jobs = n_jobs
    self.tree_method = tree_method
    self.reg_lambda = reg_lambda
    self.reg_alpha = reg_alpha
    self.subsample = subsample
    self.n_z = n_z
    self.xgboost_kwargs = xgboost_kwargs

    if model == 'random_forest' and np.sum(np.isnan(X1)) > 0:
      raise Error('The dataset must not contain missing data in order to use model=random_forest')

    self.diffusion_type = diffusion_type
    self.sde = None
    self.eps = eps
    self.beta_min = beta_min
    self.beta_max = beta_max
    if diffusion_type == 'vp':
      self.sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=n_t)

    self.n_batch = n_batch
    if self.n_batch == 0: 
      if duplicate_K > 1: # we duplicate the data multiple times, so that X0 is k times bigger so we have more room to learn
        X1 = np.tile(X1, (duplicate_K, 1))
        if X_covs is not None:
          X_covs = np.tile(X_covs, (duplicate_K, 1))

      X0 = np.random.normal(size=X1.shape) # Noise data

      # Make Datasets of interpolation
      X_train, y_train = build_data_xt(X0, X1, X_covs, n_t=self.n_t, diffusion_type=self.diffusion_type, eps=self.eps, sde=self.sde)

    if self.label_y is not None:
      assert np.sum(np.isnan(self.label_y)) == 0 # cannot have missing values in the label (just make a special categorical for nan if you need)
      self.y_uniques, self.y_probs = np.unique(self.label_y, return_counts=True)
      self.y_probs = self.y_probs/np.sum(self.y_probs)
      self.mask_y = {} # mask for which observations has a specific value of y
      for i in range(len(self.y_uniques)):
        self.mask_y[self.y_uniques[i]] = np.zeros(self.b, dtype=bool)
        self.mask_y[self.y_uniques[i]][self.label_y == self.y_uniques[i]] = True
        if self.n_batch == 0: 
          self.mask_y[self.y_uniques[i]] = np.tile(self.mask_y[self.y_uniques[i]], (duplicate_K))
    else: # assuming a single unique label 0
      self.y_probs = np.array([1.0])
      self.y_uniques = np.array([0])
      self.mask_y = {} # mask for which observations has a specific value of y
      self.mask_y[0] = np.ones(X1.shape[0], dtype=bool)

    if self.n_batch > 0: # Data iterator, no need to duplicate, not make xt yet
      rows_per_batch = self.b // self.n_batch
      batches = [rows_per_batch for i in range(self.n_batch-1)] + [self.b - rows_per_batch*(self.n_batch-1)]
      X1_splitted = {}
      X_covs_splitted = {}
      for i in self.y_uniques:
        X1_splitted[i] = np.split(X1[self.mask_y[i], :], batches, axis=0)
        if X_covs is not None:
          X_covs_splitted[i] = np.split(X_covs[self.mask_y[i], :], batches, axis=0)
        else:
          X_covs_splitted[i] = None

    # Fit model(s)
    n_steps = n_t
    n_y = len(self.y_uniques) # for each class train a seperate model
    t_levels = np.linspace(eps, 1, num=n_t)

    if self.p_in_one:
      if self.n_jobs == 1:
        self.regr = [[None for i in range(n_steps)] for j in self.y_uniques]
        for i in range(n_steps):
          for j in range(len(self.y_uniques)):
              if self.n_batch > 0: # Data iterator, no need to duplicate, not make xt yet
                self.regr[j][i] = self.train_iterator(X1_splitted[j], X_covs_splitted[j], t=t_levels[i], dim=None)
              else:
                self.regr[j][i] = self.train_parallel(
                X_train.reshape(self.n_t, self.b*self.duplicate_K, self.c_all)[i][self.mask_y[j], :], 
                y_train.reshape(self.b*self.duplicate_K, self.c)[self.mask_y[j], :]
                )
      else:
        if self.n_batch > 0: # Data iterator, no need to duplicate, not make xt yet
          self.regr = Parallel(n_jobs=self.n_jobs)(delayed(self.train_iterator)(X1_splitted[j], X_covs_splitted[j], t=t_levels[i], dim=None) for i in range(n_steps) for j in self.y_uniques)
        else:
          self.regr = Parallel(n_jobs=self.n_jobs)( # using all cpus
                  delayed(self.train_parallel)(
                    X_train.reshape(self.n_t, self.b*self.duplicate_K, self.c_all)[i][self.mask_y[j], :], 
                    y_train.reshape(self.b*self.duplicate_K, self.c)[self.mask_y[j], :]
                    ) for i in range(n_steps) for j in self.y_uniques
                  )
        # Replace fits with doubly loops to make things easier
        self.regr_ = [[None for i in range(n_steps)] for j in self.y_uniques]
        current_i = 0
        for i in range(n_steps):
          for j in range(len(self.y_uniques)): 
            self.regr_[j][i] = self.regr[current_i]
            current_i += 1
        self.regr = self.regr_
    else:
      if self.n_jobs == 1:
        self.regr = [[[None for k in range(self.c)] for i in range(n_steps)] for j in self.y_uniques]
        for i in range(n_steps):
          for j in range(len(self.y_uniques)): 
            for k in range(self.c): 
              if self.n_batch > 0: # Data iterator, no need to duplicate, not make xt yet
                self.regr[j][i][k] = self.train_iterator(X1_splitted[j], X_covs_splitted[j], t=t_levels[i], dim=k)
              else:
                self.regr[j][i][k] = self.train_parallel(
                X_train.reshape(self.n_t, self.b*self.duplicate_K, self.c_all)[i][self.mask_y[j], :], 
                y_train.reshape(self.b*self.duplicate_K, self.c)[self.mask_y[j], k]
                )
      else:
        if self.n_batch > 0: # Data iterator, no need to duplicate, not make xt yet
          self.regr = Parallel(n_jobs=self.n_jobs)(delayed(self.train_iterator)(X1_splitted[j], X_covs_splitted[j], t=t_levels[i], dim=k) for i in range(n_steps) for j in self.y_uniques for k in range(self.c))
        else:
          self.regr = Parallel(n_jobs=self.n_jobs)( # using all cpus
                  delayed(self.train_parallel)(
                    X_train.reshape(self.n_t, self.b*self.duplicate_K, self.c_all)[i][self.mask_y[j], :], 
                    y_train.reshape(self.b*self.duplicate_K, self.c)[self.mask_y[j], k]
                    ) for i in range(n_steps) for j in self.y_uniques for k in range(self.c)
                  )
        # Replace fits with doubly loops to make things easier
        self.regr_ = [[[None for k in range(self.c)] for i in range(n_steps)] for j in self.y_uniques]
        current_i = 0
        for i in range(n_steps):
          for j in range(len(self.y_uniques)): 
            for k in range(self.c): 
              self.regr_[j][i][k] = self.regr[current_i]
              current_i += 1
        self.regr = self.regr_

  def train_iterator(self, X1_splitted, X_covs_splitted, t, dim):
    np.random.seed(self.seed)

    it = IterForDMatrix(X1_splitted, X_covs_splitted, t=t, dim=dim, n_batch=self.n_batch, n_epochs=self.duplicate_K, diffusion_type=self.diffusion_type, eps=self.eps, sde=self.sde)
    data_iterator = xgb.QuantileDMatrix(it)

    xgb_dict = {'objective': 'reg:squarederror', 'eta': self.eta, 'max_depth': self.max_depth,
          "reg_lambda": self.reg_lambda, 'reg_alpha': self.reg_alpha, "subsample": self.subsample, "seed": self.seed, 
          "tree_method": self.tree_method, 'device': 'cuda' if self.gpu_hist else 'cpu', 
          "device": "cuda" if self.gpu_hist else 'cpu'}
    for myarg in self.xgboost_kwargs:
      xgb_dict[myarg] = self.xgboost_kwargs[myarg]
    out = xgb.train(xgb_dict, data_iterator, num_boost_round=self.n_estimators)

    return out

  def train_parallel(self, X_train, y_train):

    if self.model == 'random_forest':
      out = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.seed)
    elif self.model == 'lgbm':
      out = LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves, learning_rate=0.1, random_state=self.seed, force_col_wise=True)
    elif self.model == 'catboost':
      out = CatBoostRegressor(iterations=self.n_estimators, loss_function='RMSE', max_depth=self.max_depth, silent=True,
        l2_leaf_reg=0.0, random_seed=self.seed) # consider t as a golden feature if t is a variable
    elif self.model == 'xgboost':
      out = xgb.XGBRegressor(n_estimators=self.n_estimators, objective='reg:squarederror', eta=self.eta, max_depth=self.max_depth, 
        reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, subsample=self.subsample, seed=self.seed, tree_method=self.tree_method, 
        device='cuda' if self.gpu_hist else 'cpu', **self.xgboost_kwargs)
    else:
      raise Exception("model value does not exists")

    if len(y_train.shape) == 1:
      y_no_miss = ~np.isnan(y_train)
      out.fit(X_train[y_no_miss, :], y_train[y_no_miss])
    else:
      out.fit(X_train, y_train)

    return out

  def dummify(self, X):
    df = pd.DataFrame(X, columns = [str(i) for i in range(X.shape[1])]) # to Pandas
    df_names_before = df.columns
    for i in self.cat_indexes:
      df = pd.get_dummies(df, columns=[str(i)], prefix=str(i), dtype='float', drop_first=True)
    df_names_after = df.columns
    df = df.to_numpy()
    return df, df_names_before, df_names_after

  def unscale(self, X):
    if self.scaler is not None: # unscale the min-max normalization
      X = self.scaler.inverse_transform(X)
    return X

  # Rounding for the categorical variables which are dummy-coded and then remove dummy-coding
  def clean_onehot_data(self, X):
    if len(self.cat_indexes) > 0: # ex: [5, 3] and X_names_after [gender_a gender_b cartype_a cartype_b cartype_c]
      X_names_after = copy.deepcopy(self.X_names_after.to_numpy())
      prefixes = [x.split('_')[0] for x in self.X_names_after if '_' in x] # for all categorical variables, we have prefix ex: ['gender', 'gender']
      unique_prefixes = np.unique(prefixes) # uniques prefixes
      for i in range(len(unique_prefixes)):
        cat_vars_indexes = [unique_prefixes[i] + '_' in my_name for my_name in self.X_names_after]
        cat_vars_indexes = np.where(cat_vars_indexes)[0] # actual indexes
        cat_vars = X[:, cat_vars_indexes] # [b, c_cat]
        # dummy variable, so third category is true if all dummies are 0
        cat_vars = np.concatenate((np.ones((cat_vars.shape[0], 1))*0.5,cat_vars), axis=1)
        # argmax of -1, -1, 0 is 0; so as long as they are below 0 we choose the implicit-final class
        max_index = np.argmax(cat_vars, axis=1) # argmax across all the one-hot features (most likely category)
        X[:, cat_vars_indexes[0]] = max_index
        X_names_after[cat_vars_indexes[0]] = unique_prefixes[i] # gender_a -> gender
      df = pd.DataFrame(X, columns = X_names_after) # to Pandas
      df = df[self.X_names_before] # remove all gender_b, gender_c and put everything in the right order
      X = df.to_numpy()
    return X

  # Unscale and clip to prevent going beyond min-max and also round of the integers
  def clip_extremes(self, X):
    if self.int_indexes is not None:
      for i in self.int_indexes:
        X[:,i] = np.round(X[:,i], decimals=0)
    small = (X < self.X_min).astype(float)
    X = small*self.X_min + (1-small)*X
    big = (X > self.X_max).astype(float)
    X = big*self.X_max + (1-big)*X
    return X

  def predict_over_c(self, X, i, j, k, dmat, expand=False, X_covs=None):
    if X_covs is not None:
      X = np.concatenate((X, X_covs), axis=1)
    if dmat:
      X_used = xgb.DMatrix(data=X)
    else:
      X_used = X
    if k is None:
      return self.regr[j][i].predict(X_used)
    elif expand:
        return np.expand_dims(self.regr[j][i][k].predict(X_used), axis=1) # [b, 1]
    else:
      return self.regr[j][i][k].predict(X_used)

  # Return the score-fn or ode-flow output
  def my_model(self, t, y, mask_y=None, dmat=False, unflatten=True, X_covs=None):
    if unflatten:
      # y is [b*c]
      c = self.c
      b = y.shape[0] // c
      X = y.reshape(b, c) # [b, c]
    else:
      X = y

    # Output
    out = np.zeros(X.shape) # [b, c]
    i = int(round(t*(self.n_t-1)))
    for j, label in enumerate(self.y_uniques):
      if X_covs is not None:
        X_covs_masked = X_covs[mask_y[label], :]
      else:
        X_covs_masked = None
      if mask_y[label].sum() > 0:
        if self.p_in_one:
          out[mask_y[label], :] = self.predict_over_c(X=X[mask_y[label], :], i=i, j=j, k=None, dmat=dmat, X_covs=X_covs_masked)
        else:
          for k in range(self.c):
            out[mask_y[label], k] = self.predict_over_c(X=X[mask_y[label], :], i=i, j=j, k=k, dmat=dmat, X_covs=X_covs_masked)

    if self.diffusion_type == 'vp':
      alpha_, sigma_ = self.sde.marginal_prob_coef(X, t)
      out = - out / sigma_
    if unflatten:
      out = out.reshape(-1) # [b*c]
    return out

  # For imputation, we only give out and receive the missing values while ensuring consistency for the non-missing values
  # y0 is prior data, X_miss is real data
  def my_model_imputation(self, t, y, X_miss, sde=None, mask_y=None, dmat=False, X_covs=None):

    if X_covs is not None:
      assert X_covs.shape[0] == X_miss.shape[0]

    y0 = np.random.normal(size=X_miss.shape) # Noise data
    b, c = y0.shape

    if self.diffusion_type == 'vp':
      assert sde is not None
      mean, std = sde.marginal_prob(X_miss, t)
      X = mean + std*y0 # following the sde
    else:
      X = t*X_miss + (1-t)*y0 # interpolation based on ground-truth for non-missing data
    mask_miss = np.isnan(X_miss)
    X[mask_miss] = y # replace missing data by y(t)

    # Output
    out = np.zeros(X.shape) # [b, c]
    i = int(round(t*(self.n_t-1)))
    for j, label in enumerate(self.y_uniques):
      if X_covs is not None:
        X_covs_masked = X_covs[mask_y[label], :]
      else:
        X_covs_masked = None
      if mask_y[label].sum() > 0:
        if self.p_in_one:
          out[mask_y[label], :] = self.predict_over_c(X=X[mask_y[label], :], i=i, j=j, k=None, dmat=dmat, X_covs=X_covs_masked)
        else:
          for k in range(self.c):
            out[mask_y[label], k] = self.predict_over_c(X=X[mask_y[label], :], i=i, j=j, k=k, dmat=dmat, X_covs=X_covs_masked)

    if self.diffusion_type == 'vp':
      alpha_, sigma_ = self.sde.marginal_prob_coef(X, t)
      out = - out / sigma_

    out = out[mask_miss] # only return the missing data output
    out = out.reshape(-1) # [-1]
    return out

# ============================================================================
# compute_deviation_score
# ============================================================================


  def compute_deviation_score(self, test_samples, diffusion_type, n_t, duplicate_K_test=1):

    """
    - copy test sample    duplicate_K_test times -> n_samples_rep 
    - for each copy, sample a random noise -> build interpolation samples at each noise level -> x_t_samples_rep and calculate the true velocity v_true (rectified flow)
    - let the model predict the velocity of every interpolation sample at every noise level for every rep_sample
    - compute the squared error between true velocity and predicted velocity at every noise level
    - sum up the error of all noise levels for every rep_sample -> anomaly score per rep_sample
    - average over all rep_samples -> final anomaly score per test sample

    """
    assert self.diffusion_type == 'flow', "Deviation score only for flow-matching"
    assert not np.isnan(test_samples).any(), "test_samples must not contain NaNs"
    assert self.diffusion_type == diffusion_type, "Diffusion type must be the same as the trained model"
    assert n_t == self.n_t, "n_t must match training n_t"
    if self.label_y is not None:
        raise Exception("Anomaly score only for unsupervised learning")
    
    if n_t is None:
        n_t = self.n_t
    if len(self.cat_indexes) > 0:
      test_samples, column_names_before, column_names_after = self.dummify(test_samples)
    
    test_samples = self.scaler.transform(test_samples)
    
    n_samples = test_samples.shape[0]
    n_features = test_samples.shape[1]

    #duplicate test samples in form [duplicate_K_test x n_samples, n_features]
    test_samples_rep = np.tile(test_samples, (duplicate_K_test, 1))
    n_samples_rep = test_samples_rep.shape[0] 

  #create mask, for class because unsupervised (label_y = None), as in other methods
    mask_y = {0: np.ones(n_samples_rep, dtype=bool)}

  #take partial model bc of constraints
    model = partial(
        self.my_model,
        mask_y=mask_y,         
        dmat=self.n_batch > 0,
        unflatten=False,       
        X_covs=None
    )
    #for each test_sample_rep copy sample a random noise
    X0 = np.random.normal(size=test_samples_rep.shape)

    #returns for every test_rep_sample a interpolation sample at every noise level and the true velocity v_true
    x_t_samples_rep, v_true = build_data_xt(
        X0, 
        test_samples_rep, 
        n_t=n_t, 
        diffusion_type=diffusion_type, 
        eps=self.eps, 
    )
    #x_t samples rep looks like this : [sample1_1_t1, sample1_2_t1, sample1_2_t1..., sample2_1_t1... sample1_1_t2...; c]

    #initialise anomaly scores for every replicated test sample
    anomaly_scores_rep = np.zeros(n_samples_rep)
    #get noise levels as created in build_data_xt
    t_levels = np.linspace(self.eps, 1, n_t)

    # for i, t in enumerate(t_levels[n_t//2:]):
    #     #iterate over every noise level
    #     start_idx = i * n_samples_rep
    #     end_idx = (i + 1) * n_samples_rep
    #     X_t = x_t_samples_rep[start_idx:end_idx, :]
    for i, t in enumerate(t_levels[n_t//2:-1]):
        actual_time_idx = n_t // 2 + i
        start_idx = actual_time_idx * n_samples_rep
        end_idx = (actual_time_idx + 1) * n_samples_rep
        X_t = x_t_samples_rep[start_idx:end_idx, :]
        #predict velocity with the model for all test_samples_rep at noise level t
        v_pred_t = model(t=t, y=X_t)
        #calculate squared error between true velocity and predicted velocity
        squared_error = np.sum((v_true - v_pred_t) ** 2, axis=1)
        #sum the squared error for every noise level of one rep_sample
        anomaly_scores_rep += squared_error
    # average over all noise levels -> actually redundant 
    anomaly_scores_rep = anomaly_scores_rep 
    #group again for each test sample
    anomaly_scores_rep = anomaly_scores_rep.reshape(duplicate_K_test, n_samples)
    #average over all the errors of the duplicates for one test sample
    anomaly_scores = anomaly_scores_rep.mean(axis=0)  
    return anomaly_scores  

# ============================================================================
# compute_reconstruction_score
# ============================================================================

  def compute_reconstruction_score(self, test_samples, diffusion_type, n_t, duplicate_K_test=1):

    """ 
     - copy test sample duplicate_K_test times -> n_samples_rep 
    - for each copy sample a random noise -> build interpolation samples at each noise level -> x_t_samples_rep 
    - solve ode for every noise level t (for all the replicated samples) except last (not meaningful, since we are already at the goal)
    - compute with actual test sample the reconstruction error (squared error as a metric) for every reconstructed sample at every noise level
    - sum up the error of all noise levels for each rep_sample -> anomaly score per rep_sample
    - sum up over all the rep samples belonging for one test sample -> final anomaly score per test sample
    
    """
    assert self.diffusion_type == 'flow', "reconstruction score only for flow-matching"
    assert not np.isnan(test_samples).any(), "test_samples must not contain NaNs"
    assert self.diffusion_type == diffusion_type, "Diffusion type must be the same as the trained model"
    assert n_t == self.n_t, "n_t must match training n_t"

    if self.label_y is not None:
        raise Exception("Anomaly score only for unsupervised learning")
    
    if n_t is None:
        n_t = self.n_t
    if len(self.cat_indexes) > 0:
      test_samples, column_names_before, column_names_after = self.dummify(test_samples)
    test_samples = self.scaler.transform(test_samples)
    
    n_samples = test_samples.shape[0]
    n_features = test_samples.shape[1]

    #duplicate test sample lines duplikate_K times (columns stay the same)
    test_samples_rep = np.tile(test_samples, (duplicate_K_test, 1))
    n_samples_rep = test_samples_rep.shape[0]

    #convert into data space after scaler to compare with reconstruction
    #actually redundant here.. could just be compared with original samples
    test_samples_rep_unscaled = self.unscale(test_samples_rep.copy())
    test_samples_rep_unscaled = self.clean_onehot_data(test_samples_rep_unscaled)
    test_samples_rep_unscaled = self.clip_extremes(test_samples_rep_unscaled)

  #create mask, for class because unsupervised (label_y = None)
    mask_y = {0: np.ones(n_samples_rep, dtype=bool)}

  #take partial model bc of constraints
    model = partial(
        self.my_model,
        mask_y=mask_y,         
        dmat=self.n_batch > 0,
        unflatten=True,       
        X_covs=None
    )
    #for each test_samples copy, sample a random noise -> gives a better coverage of the latent space
    X0 = np.random.normal(size=test_samples_rep.shape)

    #returns for every test_rep_sample a new interpolation sample at every noise level
    x_t_samples_rep, _ = build_data_xt(
        X0, 
        test_samples_rep, 
        n_t=n_t, 
        diffusion_type=diffusion_type, 
        eps=self.eps, 
        sde=self.sde
    )
    #initialise anomaly scores for every replicated test sample
    anomaly_scores_rep = np.zeros(n_samples_rep)
    #get noise levels
    t_levels = np.linspace(self.eps, 1, n_t)

    for i, t in enumerate(t_levels[n_t//2:-1]):
        # Der tatsächliche Zeitindex in t_levels
        actual_time_idx = n_t // 2 + i
        
        # Korrekte Indizes für x_t_samples_rep
        start_idx = actual_time_idx * n_samples_rep
        end_idx = (actual_time_idx + 1) * n_samples_rep
        X_t = x_t_samples_rep[start_idx:end_idx, :]
        
        # Korrekte steps_left: von actual_time_idx bis zum letzten Index (n_t-1)
        steps_left = (n_t - 1) - actual_time_idx
        
        #self implemented!!
        ode_solved = euler_solve_from_x_t(
            x_t=X_t.reshape(-1),
            t0=t,
            my_model=model,
            steps_left=steps_left,
            n_t=n_t
        )
        #convert to data space in order to make comparable with test sample 
        solution = ode_solved.reshape(X_t.shape[0], self.c) 
        solution = self.unscale(solution)
        solution = self.clean_onehot_data(solution)
        solution = self.clip_extremes(solution)

        #calculate the squared error on every reconstructed sample to the actual test sample
        squared_error = np.sum((test_samples_rep_unscaled - solution) ** 2, axis=1)

        #sum the squared error for every noise level of one rep_sample
        anomaly_scores_rep += squared_error

    anomaly_scores_rep = anomaly_scores_rep.reshape(duplicate_K_test, n_samples) #group again for each test sample 
    #average over all rep_samples for one test sample
    anomaly_scores = anomaly_scores_rep.mean(axis=0)  #calculate mean over all duplicate_K_test copies
    return anomaly_scores  

# ============================================================================
# compute_decision_score
# ============================================================================

  def compute_decision_score(self, test_samples, diffusion_type, n_t, duplicate_K_test=1):

    """
   # same as deviation score but only considers the last noise level t = 1 , and inspired from decision function of tccm
    """
    assert self.diffusion_type == 'flow', "Decision score only for flow-matching"
    assert not np.isnan(test_samples).any(), "test_samples must not contain NaNs"
    assert self.diffusion_type == diffusion_type, "Diffusion type must be the same as the trained model"
    assert n_t == self.n_t, "n_t must match training n_t"
    if self.label_y is not None:
        raise Exception("Anomaly score only for unsupervised learning")
    
    if n_t is None:
        n_t = self.n_t
    if len(self.cat_indexes) > 0:
      test_samples, column_names_before, column_names_after = self.dummify(test_samples)
    test_samples = self.scaler.transform(test_samples)
    
    n_samples = test_samples.shape[0]
    n_features = test_samples.shape[1]

    #duplicate test samples
    test_samples_rep = np.tile(test_samples, (duplicate_K_test, 1))
    n_samples_rep = test_samples_rep.shape[0] 

  #create mask, for class because unsupervised (label_y = None)
    mask_y = {0: np.ones(n_samples_rep, dtype=bool)}

  #take partial model bc of constraints
    model = partial(
        self.my_model,
        mask_y=mask_y,         
        dmat=self.n_batch > 0,
        unflatten=False,       
        X_covs=None
    )
    #for each test_sample_rep copy sample a random noise
    X0 = np.random.normal(size=test_samples_rep.shape)

    #returns for every test_rep_sample a new interpolation sample at every noise level and the true velocity v_true
    #not really needed to get all noise levels, but we keep it for consistency with deviation score
    x_t_samples_rep, v_true = build_data_xt(
        X0, 
        test_samples_rep, 
        n_t=n_t, 
        diffusion_type=diffusion_type, 
        eps=self.eps, 
        sde=self.sde
    )
    anomaly_scores_rep = np.zeros(n_samples_rep)
    t_levels = np.linspace(self.eps, 1, n_t)

    #only the last noise level t=1 is considered
    #for all test_samples_rep get the samples at the same noise level t (here its the last one)
    start_idx = (n_t-2) * n_samples_rep
    end_idx = (n_t-1) * n_samples_rep
    X_t = x_t_samples_rep[start_idx:end_idx, :]
    #predict velocity with the model for all test_samples_rep at noise level t
    v_pred_t = model(t=t_levels[n_t-2], y=X_t)
    #calculate sum of squared error between true velocity and predicted velocity
    squared_error = np.sum((v_true - v_pred_t) ** 2, axis=1)

    #sum the squared error for every noise level of one rep_sample
    anomaly_scores_rep += squared_error

    #group again for each test sample
    anomaly_scores_rep = anomaly_scores_rep.reshape(duplicate_K_test, n_samples)
    #average over all test_samples_rep for one test sample
    anomaly_scores = anomaly_scores_rep.mean(axis=0)  

    return anomaly_scores  
  
  # ============================================================================
# compute_deviation_score_vp
# ============================================================================

  def compute_deviation_score_vp(self, test_samples, n_t, diffusion_type, duplicate_K_test=1):
      """
      Deviation Score for vp
    
      """
      assert not np.isnan(test_samples).any(), "test_samples must not contain NaNs"
      assert self.diffusion_type == 'vp', "This function is only for VP diffusion"
      assert self.diffusion_type == diffusion_type, "Diffusion type must be the same as the trained model"
      assert n_t == self.n_t, "n_t must match training n_t"
      
      if self.label_y is not None:
          raise Exception("Anomaly score only for unsupervised learning")
      
      if n_t is None:
          n_t = self.n_t
          
      if len(self.cat_indexes) > 0:
          test_samples, column_names_before, column_names_after = self.dummify(test_samples)
      test_samples = self.scaler.transform(test_samples)
      
      n_samples = test_samples.shape[0]
      n_features = test_samples.shape[1]

      # Duplicate test samples
      test_samples_rep = np.tile(test_samples, (duplicate_K_test, 1))
      n_samples_rep = test_samples_rep.shape[0]

      # Create mask for unsupervised (label_y = None)
      mask_y = {0: np.ones(n_samples_rep, dtype=bool)}

      # Partial model
      model = partial(
          self.my_model,
          mask_y=mask_y,
          dmat=self.n_batch > 0,
          unflatten=False,
          X_covs=None
      )
      
      # Sample random noise for each test_sample_rep
      X0 = np.random.normal(size=test_samples_rep.shape)
      
      anomaly_scores_rep = np.zeros(n_samples_rep)
      t_levels = np.linspace(self.eps, 1, n_t) 

      #for i, t in enumerate(t_levels[n_t//2:-1]):
      for i, t in enumerate(t_levels[1:(n_t//2)+1]):

          # Create x_t using VP forward process: x_t = mean + std * noise

          mean, std = self.sde.marginal_prob(test_samples_rep, t)
          X_t = mean + std * X0
          
          # Model predicts score function (my_model already does -eps/sigma transformation)
          score_pred = model(t=t, y=X_t)
          
          # True score function: s(x_t, t) = -noise / sigma(t)
          _, sigma_ = self.sde.marginal_prob_coef(X_t, t)
          score_true = -X0 / sigma_
          
          # Squared error between true and predicted score
          squared_error = np.sum((score_true - score_pred) ** 2, axis=1)
          anomaly_scores_rep += squared_error

      # Group again for each test sample
      anomaly_scores_rep = anomaly_scores_rep.reshape(duplicate_K_test, n_samples)
      
      # Average over all duplicate_K_test copies
      anomaly_scores = anomaly_scores_rep.mean(axis=0)
      
      return anomaly_scores


  # ============================================================================
  # compute_decision_score_vp
  # ============================================================================

  def compute_decision_score_vp(self, test_samples, n_t, diffusion_type, duplicate_K_test=1):
      """
      Decision Score for vp

      """
      assert not np.isnan(test_samples).any(), "test_samples must not contain NaNs"
      assert self.diffusion_type == 'vp', "This function is only for VP diffusion"
      assert self.diffusion_type == diffusion_type, "Diffusion type must be the same as the trained model"
      assert n_t == self.n_t, "n_t must match training n_t"

      if self.label_y is not None:
          raise Exception("Anomaly score only for unsupervised learning")
      
      if n_t is None:
          n_t = self.n_t
          
      if len(self.cat_indexes) > 0:
          test_samples, column_names_before, column_names_after = self.dummify(test_samples)
      test_samples = self.scaler.transform(test_samples)
      
      n_samples = test_samples.shape[0]
      n_features = test_samples.shape[1]

      # Duplicate test samples
      test_samples_rep = np.tile(test_samples, (duplicate_K_test, 1))
      n_samples_rep = test_samples_rep.shape[0]

      # Create mask for unsupervised (label_y = None)
      mask_y = {0: np.ones(n_samples_rep, dtype=bool)}

      # Partial model
      model = partial(
          self.my_model,
          mask_y=mask_y,
          dmat=self.n_batch > 0,
          unflatten=False,
          X_covs=None
      )
      
      # Sample random noise for each test_sample_rep
      X0 = np.random.normal(size=test_samples_rep.shape)

      t_levels = np.linspace(self.eps, 1, n_t)
      
      # Create x_t using VP forward process
      mean, std = self.sde.marginal_prob(test_samples_rep, t_levels[1])
      X_t = mean + std * X0
      
      # Model predicts score function
      score_pred = model(t=t_levels[1], y=X_t)
      
      # True score function
      _, sigma_ = self.sde.marginal_prob_coef(X_t, t_levels[1])
      score_true = -X0 / sigma_
      
      # Squared error between true and predicted score
      anomaly_scores_rep = np.sum((score_true - score_pred) ** 2, axis=1)
      
      # Group again for each test sample
      anomaly_scores_rep = anomaly_scores_rep.reshape(duplicate_K_test, n_samples)
      
      # Average over all duplicate_K_test copies
      anomaly_scores = anomaly_scores_rep.mean(axis=0)
      
      return anomaly_scores

  def compute_reconstruction_score_vp(self, test_samples, n_t, diffusion_type, duplicate_K_test=1):
      """
      Reconstruction score for VP diffusion
      
      """
      assert not np.isnan(test_samples).any(), "test_samples must not contain NaNs"
      assert self.diffusion_type == 'vp', "This function is only for VP diffusion"
      assert self.diffusion_type == diffusion_type, "Diffusion type must be the same as the trained model"
      assert n_t == self.n_t, "n_t must match training n_t"

      if self.label_y is not None:
          raise Exception("Anomaly score only for unsupervised learning")

      if n_t is None:
          n_t = self.n_t

      # Categorical handling + Scaling wie in den anderen Scores
      if len(self.cat_indexes) > 0:
          test_samples, _, _ = self.dummify(test_samples)
      test_samples = self.scaler.transform(test_samples)

      n_samples = test_samples.shape[0]
      n_features = test_samples.shape[1]

      # Dupliziere Samples für MC über Noise
      test_samples_rep = np.tile(test_samples, (duplicate_K_test, 1))
      n_samples_rep = test_samples_rep.shape[0]

      # Ground-Truth x0 im Datenraum zum Vergleich (unscaled + One-Hot-Cleaning + Clipping)
      #actually redundant here.. could just be compared with original samples below
      test_samples_rep_unscaled = self.unscale(test_samples_rep.copy())
      test_samples_rep_unscaled = self.clean_onehot_data(test_samples_rep_unscaled)
      test_samples_rep_unscaled = self.clip_extremes(test_samples_rep_unscaled)

      # Mask für unsupervised (label_y = None)
      mask_y = {0: np.ones(n_samples_rep, dtype=bool)}

      # Modell gibt Score (my_model macht schon: x0_hat -> score = -x0_hat/sigma)
      model = partial(
          self.my_model,
          mask_y=mask_y,
          dmat=self.n_batch > 0,
          unflatten=False,
          X_covs=None
      )

      # Noise z ~ N(0, I)
      X0 = np.random.normal(size=test_samples_rep.shape)

      anomaly_scores_rep = np.zeros(n_samples_rep)
      t_levels = np.linspace(self.eps, 1, n_t)

      #for t in t_levels[n_t//2:-1]:
      for t in t_levels[1:(n_t//2)+1]:

          # Vorwärts-Diffusion: x_t = mean + std * z
          mean, std = self.sde.marginal_prob(test_samples_rep, t)
          X_t = mean + std * X0     # [n_samples_rep, n_features]

          # Score-Schätzung s_theta(x_t, t)
          score_pred = model(t=t, y=X_t)   # gleiche Shape wie X_t

          # alpha(t), sigma(t) aus SDE holen
          alpha_, sigma_ = self.sde.marginal_prob_coef(test_samples_rep, t)
          # Sicherstellen, dass sie broadcastbar sind
          if np.ndim(alpha_) == 1:
              alpha_ = alpha_.reshape(-1, 1)
          if np.ndim(sigma_) == 1:
              sigma_ = sigma_.reshape(-1, 1)

          # Rekonstruktion x_hat0 im SCALED Space:
          # x_hat = (x_t + sigma^2 * score_pred) / alpha
          #tweedi formula
          x_hat_scaled = (X_t + (sigma_ ** 2) * score_pred) / alpha_

          # Zurück in Datenraum bringen
          x_hat_unscaled = self.unscale(x_hat_scaled.copy())
          x_hat_unscaled = self.clean_onehot_data(x_hat_unscaled)
          x_hat_unscaled = self.clip_extremes(x_hat_unscaled)

          # Squared Error zur echten Test-Sample-Kopie
          squared_error = np.sum((test_samples_rep_unscaled - x_hat_unscaled) ** 2, axis=1)
          anomaly_scores_rep += squared_error

      # Wieder zu [duplicate_K_test, n_samples] formen und über Noise-MC mitteln
      anomaly_scores_rep = anomaly_scores_rep.reshape(duplicate_K_test, n_samples)
      anomaly_scores = anomaly_scores_rep.mean(axis=0)

      return anomaly_scores


  # ============================================================================
  # 
  # =========================================================================

  # ============================================================================
  # 
  # =========================================================================


  # Generate new data by solving the reverse ODE/SDE
  def generate(self, batch_size=None, n_t=None, X_covs=None):

    if X_covs is not None:
      assert X_covs.shape[0] == batch_size

    # Generate prior noise
    y0 = np.random.normal(size=(self.b if batch_size is None else batch_size, self.c))

    # Generate random labels
    label_y = self.y_uniques[np.argmax(np.random.multinomial(1, self.y_probs, size=y0.shape[0]), axis=1)]
    mask_y = {} # mask for which observations has a specific value of y
    for i in range(len(self.y_uniques)):
      mask_y[self.y_uniques[i]] = np.zeros(y0.shape[0], dtype=bool)
      mask_y[self.y_uniques[i]][label_y == self.y_uniques[i]] = True
    my_model = partial(self.my_model, mask_y=mask_y, dmat=self.n_batch > 0, X_covs=X_covs)

    if self.diffusion_type == 'vp':
      sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.n_t if n_t is None else n_t)
      ode_solved = get_pc_sampler(my_model, sde=sde, denoise=True, eps=self.eps)(y0.reshape(-1))
    else:
      ode_solved = euler_solve(my_model=my_model, y0=y0.reshape(-1), N=self.n_t if n_t is None else n_t) # [t, b*c]
    solution = ode_solved.reshape(y0.shape[0], self.c) # [b, c]
    solution = self.unscale(solution)
    solution = self.clean_onehot_data(solution)
    solution = self.clip_extremes(solution)

    # Concatenate y label if needed
    if self.label_y is not None:
      solution = np.concatenate((solution, np.expand_dims(label_y, axis=1)), axis=1) 
      
    return solution

  # Impute missing data by solving the reverse ODE while keeping the non-missing data intact
  def impute(self, k=1, X=None, label_y=None, repaint=False, r=5, j=0.1, n_t=None, X_covs=None): # X is data with missing values
    assert self.diffusion_type != 'flow' # cannot use with flow=matching

    if X is None:
      X = self.X1
    if label_y is None:
      label_y = self.label_y
    if n_t is None:
      n_t = self.n_t

    if X_covs is not None:
      assert X_covs.shape[0] == X.shape[0]

    if self.diffusion_type == 'vp':
      sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=n_t)

    if label_y is None: # single category 0
      mask_y = {}
      mask_y[0] = np.ones(X.shape[0], dtype=bool)
    else:
      mask_y = {} # mask for which observations has a specific value of y
      for i in range(len(self.y_uniques)):
        mask_y[self.y_uniques[i]] = np.zeros(X.shape[0], dtype=bool)
        mask_y[self.y_uniques[i]][label_y == self.y_uniques[i]] = True

    my_model_imputation = partial(self.my_model_imputation, X_miss=X, sde=sde, mask_y=mask_y, dmat=self.n_batch > 0, X_covs=X_covs)

    for i in range(k):
      y0 = np.random.normal(size=X.shape)

      mask_miss = np.isnan(X)
      y0_miss = y0[mask_miss].reshape(-1)
      solution = copy.deepcopy(X) # Solution start with dataset which contains some missing values
      if self.diffusion_type == 'vp':
        ode_solved = get_pc_sampler(my_model_imputation, sde=sde, denoise=True, repaint=repaint)(y0_miss, r=r, j=int(math.ceil(j*n_t)))
        solution[mask_miss] = ode_solved # replace missing values with imputed values
      solution = self.unscale(solution)
      solution = self.clean_onehot_data(solution)
      solution = self.clip_extremes(solution)
      # Concatenate y label if needed
      if self.label_y is not None:
        solution = np.concatenate((solution, np.expand_dims(label_y, axis=1)), axis=1) 
      if i == 0:
        imputed_data = np.expand_dims(solution, axis=0)
      else:
        imputed_data = np.concatenate((imputed_data, np.expand_dims(solution, axis=0)), axis=0)
    return imputed_data[0] if k==1 else imputed_data

  # Zero-shot classification of one batch
  def zero_shot_classification(self, x, n_t=10, n_z=10, X_covs=None):
    assert self.label_y is not None # must have label conditioning to work
    if X_covs is not None:
      assert X_covs.shape[0] == x.shape[0]

    h = 1 / n_t
    num_classes = len(self.y_uniques)
    L2_dist = []
    for i in range(num_classes): # for each class

      # Class conditioning
      mask_y = {}
      for k in range(len(self.y_uniques)):
        if k == i:
          mask_y[self.y_uniques[k]] = np.ones(x.shape[0], dtype=bool)
        else:
          mask_y[self.y_uniques[k]] = np.zeros(x.shape[0], dtype=bool)

      L2_dist_ = []
      for k in range(n_z): # monte-carlo over multiple noises
        t = 0
        for j in range(n_t-1): # averaging over multiple noise levels [t=1/n, ... (n-1)/n]
          t = t + h
          np.random.seed(10000*k + j)
          y0 = np.random.normal(size=x.shape)
          xt = get_xt(x1=x, t=t, x0=y0, dim=None, diffusion_type=self.diffusion_type, eps=self.eps, sde=self.sde)[0]
          pred_ = self.my_model(t=t, y=xt, mask_y=mask_y, unflatten=False, dmat=self.n_batch > 0, X_covs=X_covs)
          if self.diffusion_type == 'flow':
            x0 = x - pred_ # x0 = x1 - (x1 - x0)
          elif self.diffusion_type == 'vp':
            x0 = pred_ # x0
          L2_dist_ += [np.expand_dims(np.sum((x0 - y0) ** 2, axis=1), axis=0)] # [1, b]
      L2_dist += [np.concatenate(L2_dist_, axis=0)] # [n_z*n_t, b]

    # Based on absolute
    L2_abs = []
    for i in range(num_classes): # for each class
      L2_abs += [np.expand_dims(np.mean(L2_dist[i], axis=0), axis=0)] # [1, b]
    L2_abs = np.concatenate(L2_abs, axis=0) # [c, b]
    prob_avg = softmax(-L2_abs, axis=0) # [b]
    most_likely_class_avg = np.argmin(L2_abs, axis=0) # [b]
    return self.y_uniques[most_likely_class_avg], prob_avg

  # Zero-shot classification using https://diffusion-classifier.github.io/static/docs/DiffusionClassifier.pdf
  # Return the absolute and relative accuracies
  def predict(self, X, n_t=None, n_z=None, X_covs=None):
    if n_t is None:
      n_t = self.n_t
    if n_z is None:
      n_z = self.n_z

    # Data transformation (assuming we get the raw data)
    if len(self.cat_indexes) > 0:
      X, _, _ = self.dummify(X) # dummy-coding for categorical variables
    X = self.scaler.transform(X)

    most_likely_class_avg, prob_avg = self.zero_shot_classification(X, n_t=n_t, n_z=n_z, X_covs=X_covs)

    return most_likely_class_avg

  def predict_proba(self, X, n_t=None, n_z=None, X_covs=None):
    if n_t is None:
      n_t = self.n_t
    if n_z is None:
      n_z = self.n_z

    # Data transformation (assuming we get the raw data)
    if len(self.cat_indexes) > 0:
      X, _, _ = self.dummify(X) # dummy-coding for categorical variables
    X = self.scaler.transform(X)

    most_likely_class_avg, prob_avg = self.zero_shot_classification(X, n_t=n_t, n_z=n_z, X_covs=X_covs)

    return prob_avg
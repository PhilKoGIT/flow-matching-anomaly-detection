import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from .functions import FlowMatching

# Core implementation of the Time-Conditioned Contraction Matching (TCCM) algorithm of scikit-learn API style
class TCCM:
    def __init__(self, n_features, epochs=100, learning_rate=0.001, batch_size=64):
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        self.model = FlowMatching(input_dim=n_features)

    def fit(self, X_train):
        """
        Train the TCCM
        """
        X = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.zeros(X.shape[0], dtype=torch.long).squeeze()
        train_loader = DataLoader(TensorDataset(X, y_train), batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.epochs):
            total_loss = 0
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                t = torch.rand(batch_x.shape[0], 1, device=batch_x.device)  # Sampling t, line 6 of Algorithm 1.
                f_xt = self.model(batch_x, t)  # Predict contraction vectors f(x, t) # 

                dx_dt = -batch_x
                loss = criterion(f_xt, dx_dt) # Calculate the batch loss, Equation 4.
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    #orginally implemented
    def decision_function(self, X_test):
        """
        Compute the anomaly scores of X_test
        """
        X = torch.tensor(X_test, dtype=torch.float32)
        X = X.to(next(self.model.parameters()).device)

        with torch.no_grad():
            t = torch.ones(X.shape[0], 1, device=X.device, dtype=torch.float32)  # Set t to 1
            f_xt = self.model(X, t)  # Predict contraction vectors
            anomaly_scores = torch.norm(f_xt + X, dim=1)  # compute the anomaly score, based on Equation 5.

        anomaly_scores = anomaly_scores.cpu().numpy()
        return anomaly_scores
    #----------------------------------------------------------------------------
    #----------------------------------------------------------------------------

    #Implementation of the scoring functions and helper functions

    #----------------------------------------------------------------------------
    #----------------------------------------------------------------------------
    
    def compute_deviation_score(self, X_test, n_t):
        """
        Compute the deviation score similar to the ForestDiffusion implementation. v_true is not constant but depended of the position.

        """
        X = torch.tensor(X_test, dtype=torch.float32)
        X = X.to(next(self.model.parameters()).device)
        x_t = self.build_data_xt_tccm(X_test, n_t)  # Build interpolated points x(t) for TCCM
        anomaly_scores = torch.zeros(X.shape[0], device=X.device)
        t_levels = torch.linspace(0, 1.0, n_t)

        with torch.no_grad():
            for i, t_val in enumerate(t_levels[:n_t//2]):
                actual_time_idx = i
                x_t_i = torch.tensor(x_t[actual_time_idx], dtype=torch.float32, device=X.device)
                t = torch.full((X.shape[0], 1), t_val.item(), device=X.device)
                v_pred = self.model(x_t_i, t)
                v_true = -x_t_i # True contraction vector at time t
                squared_error = torch.sum((v_true - v_pred) ** 2, dim=1)
                anomaly_scores += squared_error
        return (anomaly_scores).cpu().numpy()


    def build_data_xt_tccm(self, X, n_t):
        """
        Build interpolated points x(t). Like in Forest-Flow. This will be used to compute the scores.

        """
        b, c = X.shape
        
        t = np.linspace(0, 1.0, num=n_t)
        X_expanded = np.expand_dims(X, axis=0)        
        t_expanded = np.expand_dims(t, axis=(1, 2))   
        x_t = (1 - t_expanded) * X_expanded          
        
        return x_t


    def euler_solve_from_x_t(self, x_t, t0, steps_left, n_t):
            """
            Follow the learned flow from t_start to t=1. Same as Forest-Flow implementation.
            """
            y = x_t.clone() 
            h = 1/(n_t -1) 
            t_float = t0 

            device = y.device 

            with torch.no_grad():
                for step in range(steps_left):          
                    t_tensor = torch.full(
                        (y.shape[0], 1),           
                        t_float,                  
                        dtype=torch.float32,
                        device=device
                    )
                    y = y + h * self.model(y, t_tensor) 
                    t_float = t_float + h          
            return y

    def compute_reconstruction_score(self, X_test, n_t):
        """
        Similar principle as the reconstruction score from the Forest-Models
        For every time step t, start from x(t) and follow the flow to t=1.
        Measure the distance (squared error) to the origin as anomaly score and sum up for all starting points
        .
        """
        device = next(self.model.parameters()).device
        x_t = self.build_data_xt_tccm(X_test, n_t) 
        t_values = np.linspace(0, 1.0, num=n_t)  
        b = X_test.shape[0]
        anomaly_scores = np.zeros(b) 
        with torch.no_grad():
            for i, t_val in enumerate(t_values[:n_t//2]):
                actual_time_idx = i
                x_t_i = torch.tensor(x_t[actual_time_idx], dtype=torch.float32, device=device)
                steps_left = (n_t - 1) - actual_time_idx
                x_endpos = self.euler_solve_from_x_t(x_t_i, t0=t_val, steps_left=steps_left, n_t=n_t)
                squared_error = torch.sum((x_endpos - 0) ** 2, dim=1).cpu().numpy()
                anomaly_scores += squared_error
        anomaly_scores = anomaly_scores 
        
        return anomaly_scores

    
    #----------------------------------------------------------------------------
    #----------------------------------------------------------------------------
    #----------------------------------------------------------------------------
    #----------------------------------------------------------------------------


# The implementation of TCCM for robustness verification docked in nn.Module
class TCCMRobust(nn.Module):
    def __init__(self, n_features, epochs=100, learning_rate=0.001, batch_size=64):
        super().__init__()
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        self.model = FlowMatching(input_dim=n_features)

    def fit(self, X_train):
        """
        Train the TCCM
        """
        X = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.zeros(X.shape[0], dtype=torch.long).squeeze()
        train_loader = DataLoader(TensorDataset(X, y_train), batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.epochs):
            total_loss = 0
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                t = torch.rand(batch_x.shape[0], 1, device=batch_x.device)
                f_xt = self.model(batch_x, t)

                dx_dt = -batch_x
                loss = criterion(f_xt, dx_dt)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    
    def decision_function(self, X_test):
        """
        Compute the anomaly scores of X_test
        """
        X = torch.tensor(X_test, dtype=torch.float32)
        X = X.to(next(self.model.parameters()).device)

        with torch.no_grad():
            t = torch.ones(X.shape[0], 1, device=X.device, dtype=torch.float32)
            f_xt = self.model(X, t)
            anomaly_scores = torch.norm(f_xt + X, dim=1)

        anomaly_scores = anomaly_scores.cpu().numpy()
        return anomaly_scores
    
    def forward(self, x):
        t = torch.ones(x.shape[0], 1, device=x.device, dtype=torch.float32)
        f_xt = self.model(x, t)
        anomaly_scores = torch.norm(f_xt + x, dim=1)
        return anomaly_scores
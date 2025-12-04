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
    
    # added scoring function 

    #----------------------------------------------------------------------------

    def compute_deviation_score(self, X_test, n_t):
        X = torch.tensor(X_test, dtype=torch.float32)
        X = X.to(next(self.model.parameters()).device)
        
        anomaly_scores = torch.zeros(X.shape[0], device=X.device)
        t_levels = torch.linspace(0.1, 1.0, n_t)
    
        with torch.no_grad():
            for t_val in t_levels:
                t = torch.full((X.shape[0], 1), t_val.item(), device=X.device)
                v_pred = self.model(X, t)
                v_true = -X
                squared_error = torch.sum((v_true - v_pred) ** 2, dim=1)
                anomaly_scores += squared_error
        
        return (anomaly_scores / n_t).cpu().numpy()
    
    #----------------------------------------------------------------------------
 
    # added scoring function 

    #----------------------------------------------------------------------------

    #just let the point from the original follow the flow and measure the distance at the end
    def compute_reconstruction_score(self, X_test, n_t):
        """
        Reconstruction Score 
        """
        X = torch.tensor(X_test, dtype=torch.float32)
        X = X.to(next(self.model.parameters()).device)
        
        x_current = X.clone()
        dt = 1.0 / n_t
        
        with torch.no_grad():
            for step in range(n_t):
                t_val = step * dt
                #create time point tensor
                t = torch.full((X.shape[0], 1), t_val, device=X.device, dtype=torch.float32)
                v = self.model(x_current, t)
                #calculate the new pos
                x_current = x_current + v * dt
        
        anomaly_scores = torch.norm(x_current, dim=1)
        
        return anomaly_scores.cpu().numpy()

    





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
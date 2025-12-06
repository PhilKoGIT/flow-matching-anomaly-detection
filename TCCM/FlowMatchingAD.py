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

    # added scoring function 



    #SQUARED ERROR FOR CONSISTENCY WITH FOREST FLOW... was unterschied norm?!
    #----------------------------------------------------------------------------

    def build_data_xt_tccm(self, X, n_t):
        """
        Build interpolated points x(t) for TCCM. Like in Forest-Flow. This will be used to compute the reconstruction score starting from every interpolation point and summing them up.
        Returns:
            x_t: Interpolated points [n_t, b, c]
            t_values: Time points [n_t]
        """
        b, c = X.shape
        
        t = np.linspace(0, 1.0, num=n_t)
        X_expanded = np.expand_dims(X, axis=0)        # [1, b, c]
        t_expanded = np.expand_dims(t, axis=(1, 2))   # [n_t, 1, 1]
        
        x_t = (1 - t_expanded) * X_expanded           # [n_t, b, c]
        
        return x_t, t


    def follow_flow(self, x_start, t_start, n_steps):
        """
        Follow the learned flow from t_start to t=1.
        """
        x_current = x_start.clone()
        delta_t = (1.0 - t_start) / max(n_steps, 1)
        
        with torch.no_grad():
            for step in range(n_steps):
                t_val = t_start + step * delta_t
                t = torch.full((x_current.shape[0], 1), t_val, device=x_current.device)
                v = self.model(x_current, t)
                x_current = x_current + v * delta_t
        
        return x_current


    def compute_reconstruction_score(self, X_test, n_t):
        """
        Same principle as the reconstruction score from the Forest-Flow
        For every time step t, start from x(t) and follow the flow to t=1.
        Measure the distance to the origin as anomaly score and sum up the error for all starting points
        .
        """
        device = next(self.model.parameters()).device
        
        # Build interpolated points
        x_t_interpolations, t_values = self.build_data_xt_tccm(X_test, n_t)  # [n_t, b, c]
        
        b = X_test.shape[0]
        anomaly_scores = np.zeros(b)
        
        with torch.no_grad():
            #only go to n_t-1 because starting from t=1 makes no sense
            for i, t_val in enumerate(t_values[:-1]):
                # x(t) for this time point
                x_t = torch.tensor(x_t_interpolations[i], dtype=torch.float32, device=device)  # [b, c]
                
                # Follow the flow from t to 1
                steps_left = n_t - i - 1
                x_final = self.follow_flow(x_t, t_start=t_val, n_steps=steps_left)
                
                # Distance to the origin
                dist = torch.norm(x_final, dim=1).cpu().numpy()  # [b]
                anomaly_scores += dist
        
        # Average over all time points
        anomaly_scores = anomaly_scores / n_t
        
        return anomaly_scores


    def compute_simple_reconstruction_score(self, X_test, n_t):
        """
        Starts from x(0) = X_test and follows the flow to t=1 directly.
        Measures the distance to the origin as anomaly score.
        
        """
        X = torch.tensor(X_test, dtype=torch.float32)
        X = X.to(next(self.model.parameters()).device)
        
        x_current = X.clone()
        delta_t = 1.0 / n_t
        
        with torch.no_grad():
            for step in range(n_t):
                t = torch.full((X.shape[0], 1), step * delta_t, device=X.device)
                v = self.model(x_current, t)
                x_current = x_current + v * delta_t
        
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
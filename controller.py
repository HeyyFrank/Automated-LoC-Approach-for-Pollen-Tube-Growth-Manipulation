import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel 


# Kalman Filter class
class KalmanFilter:
    def __init__(self, A, B, C, Q, R, initial_state_estimate, initial_state_covariance):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.state_estimate = initial_state_estimate
        self.state_covariance = initial_state_covariance

    def update(self, control_input, observation):
        # Prediction step
        predicted_state = self.A @ self.state_estimate + self.B @ control_input
        predicted_covariance = self.A @ self.state_covariance @ self.A.T + self.Q

        # Update step
        kalman_gain = predicted_covariance @ self.C.T @ np.linalg.inv(self.C @ predicted_covariance @ self.C.T + self.R)
        self.state_estimate = predicted_state + kalman_gain @ (observation - self.C @ predicted_state)
        self.state_covariance = (np.eye(self.A.shape[0]) - kalman_gain @ self.C) @ predicted_covariance

        return self.state_estimate
    
    def update_model(self, new_B, new_Q=None, new_R=None):
        self.B = new_B
        if new_Q is not None:
            self.Q = new_Q
        if new_R is not None:
            self.R = new_R

# PID Controller class
class PIDController:
    def __init__(self, kp, ki, kd, setpoint, kalman_filter):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.kalman_filter = kalman_filter
        self.prev_error = np.zeros((2, 1))
        self.integral = np.zeros((2, 1))
        # Initialize Gaussian Process regression
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
    def train_gp(self, input_data, output_data):
        # Train Gaussian Process with input-output data
        self.gp.fit(input_data, output_data)

    def predict_gp(self, input_data):
        # Predict using Gaussian Process
        y_pred, sigma = self.gp.predict(input_data, return_std=True)
        return y_pred, sigma
    
    
    def compute(self, observation, control_input, setpoint=None):
        # # Train Gaussian Process with input-output data
        # input_data = np.concatenate([control_input, process_variable], axis=1)
        # output_data = process_variable
        # self.train_gp(input_data, output_data)
        
        # Use the Kalman filter to estimate the state
        state_estimate = self.kalman_filter.update(control_input, observation)

        # Compute the error
        if setpoint is not None:
            self.setpoint = setpoint.copy()
        error = self.setpoint - state_estimate

        # Proportional term
        P = self.kp * error

        # Integral term
        self.integral += error
        I = self.ki * self.integral

        # Derivative term
        D = self.kd * (error - self.prev_error)

        # Compute the control output
        control_output = P + I + D

        # Update previous error for the next iteration
        self.prev_error = error

        return control_output


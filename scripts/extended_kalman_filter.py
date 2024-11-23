import numpy as np

class ExtendedKalmanFilter():
    def __init__(self, process_noise, measurement_noise_covered, measurement_noise_uncovered):
        self.process_noise = process_noise
        self.measurement_noise_covered = measurement_noise_covered
        self.measurement_noise_uncovered = measurement_noise_uncovered
        self.measurement_noise = None
        self.state = None
        self.P = None
    
    def initialize_state(self, state):
        self.state = state.copy() # [x, y, theta, u, v]
        # Initialize the covariance matrix
        self.P = np.eye(len(state))/10.0
    
    def set_mode(self, covered):
        self.measurement_noise = self.measurement_noise_covered if covered else self.measurement_noise_uncovered
    
    def _state_transition(self, control_input, dt):
        # Control input: [v, w]
        # State transition function
        v, w = control_input
        theta = self.state[2]
        
        self.state[0] += v * np.cos(theta) * dt
        self.state[1] += v * np.sin(theta) * dt
        self.state[2] += w * dt
        self.state[3] = v
        self.state[4] = w
        
        # Normalize the angle
        self.state[2] = (self.state[2] + np.pi) % (2*np.pi) - np.pi
 
        
    def predict(self, control_input, dt):
        
         # Calculate Jacobian of state transition function
        v = control_input[0]
        theta = self.state[2]
        F = np.array([
            [1, 0, -v * np.sin(theta) * dt, 0, 0],
            [0, 1,  v * np.cos(theta) * dt, 0, 0], 
            [0, 0,  1, 0, 0],
            [0, 0,  0, 1, 0],
            [0, 0,  0, 0, 1]
            ])
        
        # Predict state and covariance
        self._state_transition(control_input, dt)
        self.P = F @ self.P @ F.T + self.process_noise
 
    
    def update(self, measurement):
        
        H = np.eye(len(self.state))  # Measurement model Jacobian
        
        # Calculate innovation and its covariance
        y = measurement - self.state
        
        S = H @ self.P @ H.T + self.measurement_noise
        
        # Calculate Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ y
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P
        
        # Normalize angle
        self.state[2] = (self.state[2] + np.pi) % (2*np.pi) - np.pi 
    
    @property
    def get_state(self):
        return self.state.copy() if self.state is not None else None
    
    @property
    def get_covariance(self):
        return self.P.copy() if self.P is not None else None

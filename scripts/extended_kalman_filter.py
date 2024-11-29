import numpy as np

# Values obtain in run_tests.ipynb
Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1])
R_COVERED = np.diag([9999999, 9999999, 9999999, 71.4096, 246.1112]) # No measurement from camera
R_UNCOVERED = np.diag([0.08497380, 0.11697213, 0.00000717, 71.4096, 246.1112])

class ExtendedKalmanFilter:
    def __init__(self, dt, wheel_base):

        self.dt = dt
        self.wheel_base = wheel_base
        # State vector dimension
        self.n = 5 # [x, y, theta, v, omega]
        # Control input dimension
        self.m = 2 # [v_l, v_r]
        
        # Initialize state covariance matrix
        self.P = np.eye(self.n) * 0.1
        
        # Process noise covariance
        self.Q = Q
        
        # Measurement noise covariance
        self.R = R_COVERED
        
        # Initialize state vector
        self.state = np.zeros(self.n)
        
        print("EKF initialized correctly.")
        
    #------------------------------------------#
    
    def _compute_velocity(self, v_l, v_r):
        # Compute linear and angular velocities from wheel speeds
        v = (v_r + v_l) / 2
        omega = (v_r - v_l) / self.wheel_base
        return v, omega
    #------------------------------------------#
    
    def initialize_state(self, state):
        if len(state) != self.n:
            raise ValueError(f"State vector must have length {self.n}")
        self.state = state
        
    def set_mode(self, covered):
        if covered:
            self.R = R_COVERED # No measurement from camera
        else:
            self.R = R_UNCOVERED # Measurement from camera
            
    def predict(self, u):
        if len(u) != self.m:
            raise ValueError(f"Control input must have length {self.m}")
        
        # Extract states
        x, y, theta, _, _ = self.state
        v_l, v_r = u
        
        v, omega = self._compute_velocity(v_l, v_r)
        
        # Predict next state using motion model
        x_next = x + v * np.cos(theta) * self.dt
        y_next = y + v * np.sin(theta) * self.dt
        theta_next = theta + omega * self.dt
        v_next = v
        omega_next = omega
        
        self.state = np.array([x_next, y_next, theta_next, v_next, omega_next])
        
        # Compute Jacobian of state transition
        F = np.eye(self.n)
        F[0, 2] = -v * np.sin(theta) * self.dt
        F[0, 3] = np.cos(theta) * self.dt
        F[1, 2] = v * np.cos(theta) * self.dt
        F[1, 3] = np.sin(theta) * self.dt
        F[2, 4] = self.dt
        
        # Update covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        
        measurement[3], measurement[4] = self._compute_velocity(measurement[3], measurement[4])
        
        # Measurement matrix 
        H = np.eye(self.n)
        
        # Compute Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        y = measurement - self.state
        # Normalize angle difference
        y[2] = np.arctan2(np.sin(y[2]), np.cos(y[2]))
        
        self.state = self.state + K @ y
        self.P = (np.eye(self.n) - K @ H) @ self.P
    
    def get_state_and_covariance(self):
        return self.state, self.P
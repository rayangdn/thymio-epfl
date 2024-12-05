# Extended Kalman Filter implementation for robot localization
import numpy as np

# Experimentally determined process noise covariance for state variables [x, y, θ, v, ω]
Q = np.diag([27.7276, 27.7276, 0.0554, 0.1026, 0.01])

# Measurement noise covariance when camera is covered/uncovered
R_COVERED = np.diag([9999999, 9999999, 9999999, 32.1189, 122.2820])   # High uncertainty in position/orientation
R_UNCOVERED = np.diag([0.21232803, 0.21232803, 0.00001523, 32.1189, 122.2820])  # Normal camera measurements

class ExtendedKalmanFilter:
    def __init__(self, dt, wheel_base):
        # Time step and robot physical parameters
        self.dt = dt
        self.wheel_base = wheel_base
        
        # State vector [x, y, θ, v, ω]
        self.n = 5
        # Control input [v_left, v_right]
        self.m = 2
        
        # Initialize state covariance with small uncertainty
        self.P = np.eye(self.n) * 0.1
        
        # Set noise covariance matrices
        self.Q = Q  # Process noise
        self.R = R_UNCOVERED  # Measurement noise (default: camera uncovered)
        
        # Initialize state estimate
        self.state = np.zeros(self.n)
        
        print("EKF initialized correctly.")
    
    def _compute_velocity(self, v_l, v_r):
        # Convert differential drive velocities to linear and angular velocities
        v = (v_l + v_r) / 2  # Linear velocity
        omega = (v_l - v_r) / self.wheel_base  # Angular velocity
        return v, omega
    
    def initialize_state(self, state):
        # Set initial state estimate
        self.state = state
        
    def set_mode(self, covered):
        # Switch measurement noise based on camera visibility
        self.R = R_COVERED if covered else R_UNCOVERED
            
    def predict(self, u):

        # Extract current state
        x, y, theta, _, _ = self.state
        v_l, v_r = u
        
        # Compute robot velocities from wheel speeds
        v, omega = self._compute_velocity(v_l, v_r)
        
        # Predict next state using nonlinear motion model
        x_next = x + v * np.cos(theta) * self.dt
        y_next = y + v * np.sin(theta) * self.dt
        theta_next = theta + omega * self.dt
        v_next = v
        omega_next = omega
        
        self.state = np.array([x_next, y_next, theta_next, v_next, omega_next])
        
        # Compute Jacobian of motion model
        F = np.eye(self.n)
        F[0, 2] = -v * np.sin(theta) * self.dt  # ∂x/∂θ
        F[0, 3] = np.cos(theta) * self.dt       # ∂x/∂v
        F[1, 2] = v * np.cos(theta) * self.dt   # ∂y/∂θ
        F[1, 3] = np.sin(theta) * self.dt       # ∂y/∂v
        F[2, 4] = self.dt                       # ∂θ/∂ω
        
        # Update state covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        
        # Convert wheel velocities to robot velocities
        measurement[3], measurement[4] = self._compute_velocity(measurement[3], measurement[4])
        
        # Linear measurement model
        H = np.eye(self.n)
        
        # Compute optimal Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Calculate measurement residual
        y = measurement - self.state
        
        # Normalize angle difference to [-π, π]
        y[2] = np.arctan2(np.sin(y[2]), np.cos(y[2]))
        
        # Update state estimate and covariance
        self.state = self.state + K @ y
        self.P = (np.eye(self.n) - K @ H) @ self.P
    
    def get_state_and_covariance(self):
        # Return current state estimate and uncertainty
        return self.state, self.P
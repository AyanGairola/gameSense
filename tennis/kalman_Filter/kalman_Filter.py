import numpy as np

class KalmanFilter:
    def __init__(self, initial_state):
        self.state = np.array(initial_state, dtype=np.float32)
        self.prediction = self.state
        self.prediction_covariance = np.eye(4, dtype=np.float32) * 0.1
        self.observation_covariance = np.eye(2, dtype=np.float32) * 0.1
        self.transition_matrix = np.eye(4, dtype=np.float32)
        self.transition_matrix[0, 2] = 1
        self.transition_matrix[1, 3] = 1

    def predict(self):
        self.prediction = np.dot(self.transition_matrix, self.state)
        self.prediction_covariance = np.dot(np.dot(self.transition_matrix, self.prediction_covariance), self.transition_matrix.T) + 0.1 * np.eye(4)
        return self.prediction[:2]

    def update(self, measurement):
        innovation = measurement - np.dot(np.eye(2, 4), self.prediction)
        innovation_covariance = self.observation_covariance + np.dot(np.dot(np.eye(2, 4), self.prediction_covariance), np.eye(2, 4).T)
        kalman_gain = np.dot(np.dot(self.prediction_covariance, np.eye(2, 4).T), np.linalg.inv(innovation_covariance))
        self.state = self.prediction + np.dot(kalman_gain, innovation)
        self.prediction_covariance = self.prediction_covariance - np.dot(np.dot(kalman_gain, innovation_covariance), kalman_gain.T)
        return self.state[:2]
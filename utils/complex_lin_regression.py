import numpy as np
import cmath

class complex_linear_regression:
    def fit(self, W, Z):
        W = W.ravel()
        Z = Z.ravel()
        
        X = self.design(W)
        X_star = X.conj().T
        Z = Z.reshape(-1, 1)
        
        self.beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_star, X)), X_star), Z)
        
    def predict(self, W):
        W = W.ravel()
        X = self.design(W)
        pred = np.matmul(self.beta.T, X.T).ravel()   

        return pred
    
    def residual(self, W, Z):
        pred = self.predict(W)
            
        polar = np.vectorize(cmath.polar)(Z - pred)
        polar = polar[0]
        return np.sum(polar)/len(W)

    def design(self, W):
        W = W.ravel()
        return np.c_[np.ones(len(W)), W.reshape(-1, 1)]
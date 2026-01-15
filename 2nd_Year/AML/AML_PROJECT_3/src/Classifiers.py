import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.kernel_approximation import RBFSampler
from .Utils import *

class KDENaiveBayes(BaseEstimator, ClassifierMixin):
    """
    A Naive Bayes classifier that estimates the feature probabilities using 
    Kernel Density Estimation (KDE) instead of assuming a Gaussian distribution.

    This is useful when feature distributions are multimodal or non-normal.
    
    Parameters
    ----------
    bandwidth : str or float, default='silverman'
        The bandwidth of the kernel. If 'silverman', it is estimated using 
        Silverman's rule of thumb. Otherwise, a fixed float value is expected.
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-random number generation for reproducibility.
    """

    def __init__(self, bandwidth='silverman', random_state=None):
        self.bandwidth = bandwidth
        self.random_state = random_state
        self.classes_ = None
        self.models_ = {} 
        self.priors_ = {}

    def fit(self, X, y):
        """
        Fit the KDENaiveBayes classifier according to X, y.

        For each class and each feature, a separate KernelDensity model is fitted.
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        for c in self.classes_:
            # Isolate data for the current class
            X_c = X[y == c]
            
            # Handle edge case: Class exists in `classes_` but has no samples in subset
            if len(X_c) == 0:
                self.priors_[c] = 0.0
                continue
                
            # Calculate class prior P(y)
            self.priors_[c] = len(X_c) / n_samples
            self.models_[c] = {}
            
            # 
            # We fit a separate KDE for every feature to maintain the "Naive" 
            # independence assumption: P(x|y) = \prod P(x_i|y)
            for feature_idx in range(n_features):
                feature_data = X_c[:, feature_idx]
                
                # Determine Bandwidth
                if self.bandwidth == 'silverman':
                    bw = silverman_bandwidth(feature_data)
                    # Protect against numerical instability if variance is effectively zero
                    if bw <= 1e-9: bw = 1.0 
                else:
                    bw = self.bandwidth
                
                # Note: KernelDensity is deterministic in this context
                kde = KernelDensity(bandwidth=bw, kernel='gaussian')
                kde.fit(feature_data.reshape(-1, 1))
                self.models_[c][feature_idx] = kde
        return self

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            # Add Log Prior: log(P(y))
            if self.priors_[c] > 0:
                log_probs[:, i] = np.log(self.priors_[c])
            else:
                log_probs[:, i] = -1000.0 # Effectively zero probability
            
            # Add Log Likelihoods: + \sum log(P(x_i|y))
            for feature_idx in range(n_features):
                kde = self.models_[c][feature_idx]
                log_density = kde.score_samples(X[:, feature_idx].reshape(-1, 1))
                
                # Clip values to prevent floating point overflow/underflow 
                # during the subsequent exponentiation step.
                log_density = np.clip(log_density, -700, 700)
                log_probs[:, i] += log_density

        # Normalize log-probabilities using the Log-Sum-Exp trick 
        # to ensure numerical stability: P(y|x) = exp(log_prob - log_sum_exp(log_probs))
        lse = logsumexp(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - lse)
        
        # Handle potential NaNs (e.g., if all log_probs were -inf)
        if np.isnan(probs).any():
            probs = np.nan_to_num(probs, nan=1.0/n_classes)
            
        return probs

    def predict(self, X):
        """Perform classification on an array of test vectors X."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

class RFFNaiveBayes(ClassifierMixin, BaseEstimator):
    """
    Naive Bayes that estimates feature densities using Random Fourier Features (RFF).
    
    This approximates the Kernel Density estimate without storing all training points.
    It relies on the property that the PDF estimate is the dot product of the 
    transformed test point and the "Mean Embedding" of the training points in RFF space.
    
    Math: \hat{f}(x) \approx \phi(x)^T \cdot \left( \frac{1}{N} \sum \phi(x_i) \right)

    Parameters
    ----------
    n_components : int, default=100
        The number of Monte Carlo samples (dimensions) used to approximate the RBF kernel.
        Higher values improve accuracy but increase computational cost.
    random_state : int, default=42
        Seed for the random weights in RFF.
    """

    def __init__(self, n_components=100, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        self.estimators_ = {}
        self.priors_ = {}
        
        for c in self.classes_:
            X_c = X[y == c]
            
            # 1. Calculate Priors
            if len(X_c) == 0:
                self.priors_[c] = 0.0
                continue
            self.priors_[c] = len(X_c) / n_samples
            
            self.estimators_[c] = {}
            
            # 2. Fit RFF Sampler per feature (Independence Assumption)
            for feature_idx in range(n_features):
                feature_data = X_c[:, feature_idx].reshape(-1, 1)

                # We still need a bandwidth to define the RBF Kernel's shape (gamma)
                bw = silverman_bandwidth(feature_data)

                # Convert bandwidth to gamma: gamma = 1 / (2 * sigma^2)
                gamma = 1.0 / (2 * bw**2)
                
                # Initialize RFF Sampler
                # Critical: We use a distinct random state per feature.
                # If we used the same state, features might become artificially correlated.
                rff = RBFSampler(gamma=gamma, 
                                 n_components=self.n_components, 
                                 random_state=self.random_state + feature_idx)
                
                # Transform Training Data -> High Dimensional Z-Space
                Z = rff.fit_transform(feature_data)
                
                # COMPUTE MEAN EMBEDDING (The "Distribution")
                # Instead of storing N points, we compress the entire distribution
                # into a single vector of size D (n_components).
                mean_vector = np.mean(Z, axis=0)
                
                self.estimators_[c][feature_idx] = (rff, mean_vector)
                
        return self

    def predict_proba(self, X):
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # Initialize with -inf so strict logsumexp doesn't fail on empty classes
        log_probs = np.full((n_samples, n_classes), -1000.0)
        
        for i, c in enumerate(self.classes_):
            if self.priors_[c] == 0:
                continue

            # Log Prior
            log_probs[:, i] = np.log(self.priors_[c])
            
            for feature_idx in range(n_features):
                if feature_idx not in self.estimators_[c]:
                    continue
                    
                rff, mean_vector = self.estimators_[c][feature_idx]
                
                # Transform Test Data using the fitted sampler
                Z_test = rff.transform(X[:, feature_idx].reshape(-1, 1))
                
                # ESTIMATE DENSITY via DOT PRODUCT
                # The density at x is approx. the dot product of Z(x) and the Mean Vector.
                density = np.dot(Z_test, mean_vector)
                
                # RFF approximation can technically dip slightly below 0 due to 
                # cosine nature and finite sampling noise.
                # Clamp to epsilon to avoid log(negative) errors.
                density = np.clip(density, 1e-9, None)
                
                log_probs[:, i] += np.log(density)

        # Robust Softmax (Log-Sum-Exp)
        lse = logsumexp(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - lse)
        
        return probs

    def predict(self, X):
        """Predict the most likely class."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

if __name__ == "__main__":
    pass
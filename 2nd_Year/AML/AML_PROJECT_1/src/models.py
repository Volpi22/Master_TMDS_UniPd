##################################### IMPORTS #################################################

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC as SVMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KernelDensity
from typing import List, Union
import time
from scipy.stats import iqr
from scipy import stats

from src.data_ops import data_transformation, data_standardization

##################################### COLUMN DEFINITIONS #########################################

# Define categorical columns and their data types
categorical_cols = {
        'cp': 'category',
        'restecg': 'category',
        'thal': 'category',
        'slope': 'category',
        'ca': 'category',
    }

# Define numerical columns and their data types
numerical_cols = {
        'trestbps': 'float64',
        'chol': 'float64',
        'thalach': 'float64',
        'age': 'float64',
    }

# Define special columns that require unique handling
special_cols = {
        'oldpeak': 'float64',
}

binary_cols = {
        'sex': 'int64',
        'fbs': 'int64',
        'exang': 'int64',
}

########################## HELPER FUNCTIONS ################################################

def find_optimal_threshold(y_true, y_pred_proba):
    """
    Find the optimal classification threshold using Youden's J statistic.
    
    The optimal threshold maximizes the Youden's J statistic (sensitivity + specificity - 1),
    which is equivalent to maximizing the distance from the ROC curve to the diagonal.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities for the positive class
    
    Returns:
    --------
    float
        Optimal threshold value that maximizes Youden's J statistic
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    # Youden's J statistic = sensitivity + specificity - 1 = tpr - fpr
    j_scores = tpr - fpr
    # Find the threshold that maximizes J
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

########################## MULTIVARIATE NAIVE BAYES CLASSIFIER #################################

def silverman_bandwidth(X):
    """Silverman's rule of thumb for bandwidth selection"""
    n = len(X)
    sigma = np.std(X, ddof=1)
    iqr_val = iqr(X)
    
    # Silverman's rule
    h = 0.9 * min(sigma, iqr_val / 1.34) * (n ** (-1/5))
    return h

class CustomNaiveBayesMod:
    """
    Custom Naive Bayes classifier with mixed feature types.
    
    This classifier handles:
    - Categorical features (with Laplace smoothing)
    - Binary features (using Bernoulli distribution)
    - Oldpeak feature (using Kernel Density Estimation with Gaussian kernel)
    - Numerical features including age (using multivariate Gaussian distribution)
    """
    
    def __init__(self, feature_names: List[str], categorical_cols: List[str], binary_cols: List[str], numerical_gaussian_cols: List[str], oldpeak_col: str):
        """
        Initialize the Custom Naive Bayes classifier.
        
        Parameters:
        -----------
        feature_names : List[str]
            List of all feature names
        categorical_cols : List[str]
            List of categorical column names
        binary_cols : List[str]
            List of binary column names (0/1 values)
        numerical_gaussian_cols : List[str]
            List of numerical columns (including age) to model with multivariate Gaussian
        oldpeak_col : str
            Name of the oldpeak column (to use KDE)
        """
        self.feature_names = feature_names
        self.categorical_cols = categorical_cols
        self.binary_cols = binary_cols
        self.numerical_gaussian_cols = numerical_gaussian_cols
        self.oldpeak_col = oldpeak_col
        
        # Model parameters (fitted during training)
        self.classes_ = None  # Unique class labels
        self.class_prior_ = None  # Prior probabilities P(y) for each class

        self.cat_prob_ = {}       # Conditional probabilities P(x|y) for categorical features
        self.bernoulli_prob_ = {} # Bernoulli parameters P(x=1|y) for binary features
        self.kde_ = {}             # Kernel Density Estimators for oldpeak per class
        self.gaussian_params_ = {}# Mean vectors and covariance matrices for multivariate Gaussians

    def _to_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Convert input to DataFrame if it's a numpy array.
        
        Parameters:
        -----------
        X : Union[pd.DataFrame, np.ndarray]
            Input data (DataFrame or ndarray)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with appropriate column names
        """
        if isinstance(X, pd.DataFrame):
            return X
        else:
            return pd.DataFrame(X, columns=self.feature_names)
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Fit the Naive Bayes classifier to training data.
        
        Parameters:
        -----------
        X : Union[pd.DataFrame, np.ndarray]
            Training features
        y : Union[pd.Series, np.ndarray]
            Training labels
            
        Returns:
        --------
        self : CustomNaiveBayesMod
            Fitted classifier instance
        """
        # Convert inputs to appropriate formats
        X = self._to_dataframe(X)
        y = pd.Series(y)
        
        # Calculate class priors P(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_prior_ = counts / len(y)
        
        # Fit categorical features
        for col in self.categorical_cols:
            self.cat_prob_[col] = {}
            
            # Calculate conditional probabilities P(x|y) for each class
            for cls in self.classes_:
                mask = (y == cls)
                counts_cls = X[col][mask].value_counts()
                # Apply Laplace smoothing: add 1 to numerator and number of categories to denominator
                freqs = (counts_cls + 1) / (mask.sum() + len(counts_cls))
                self.cat_prob_[col][cls] = freqs
        
        # Fit Bernoulli distribution for binary features
        # For each binary feature, estimate P(x=1|y) for each class
        self.bernoulli_prob_ = {}
        for col in self.binary_cols:
            self.bernoulli_prob_[col] = {}
            for cls in self.classes_:
                mask = (y == cls)
                # Calculate probability of feature being 1 given class, with Laplace smoothing
                # P(x=1|y) = (count(x=1, y) + 1) / (count(y) + 2)
                prob_1 = (X.loc[mask, col].sum() + 1) / (mask.sum() + 2)
                self.bernoulli_prob_[col][cls] = prob_1
        
        # Fit KDE for oldpeak feature (one KDE per class)
        self.kde_ = {}
        for cls in self.classes_:
            mask = (y == cls)
            oldpeak_vals = X.loc[mask, self.oldpeak_col].values.reshape(-1, 1)
            h = silverman_bandwidth(oldpeak_vals)
            kde = KernelDensity(kernel='gaussian', bandwidth=h)
            kde.fit(oldpeak_vals)
            self.kde_[cls] = kde
        
        # Fit multivariate Gaussian for numerical features
        self.gaussian_params_ = {}
        for cls in self.classes_:
            mask = (y == cls)
            data = X.loc[mask, self.numerical_gaussian_cols].values
            # Calculate mean vector
            mu = data.mean(axis=0)
            # Calculate covariance matrix with regularization (add small constant to diagonal)
            sigma = np.cov(data, rowvar=False) + 1e-6 * np.eye(len(self.numerical_gaussian_cols))
            sigma_inv = np.linalg.inv(sigma)
            sigma_det = np.linalg.det(sigma)
            self.gaussian_params_[cls] = {'mu': mu, 'sigma': sigma, 'sigma_inv': sigma_inv, 'sigma_det': sigma_det}
        return self
    
    def _log_gaussian_pdf(self, x, mu, sigma_inv, sigma_det):
        """
        Calculate log probability density function of multivariate Gaussian distribution.
        
        Parameters:
        -----------
        x : np.ndarray
            Data point
        mu : np.ndarray
            Mean vector
        sigma_inv : np.ndarray
            Inverse covariance matrix
        sigma_det : float
            Determinant of covariance matrix
            
        Returns:
        --------
        float
            Log probability density
        """
        d = len(mu)
        x_m = x - mu
        # Log of multivariate Gaussian PDF: log(N(x|μ,Σ))
        return -0.5 * (np.log((2*np.pi)**d * sigma_det) + np.dot(np.dot(x_m, sigma_inv), x_m))
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for input samples.
        
        Parameters:
        -----------
        X : Union[pd.DataFrame, np.ndarray]
            Input features
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_samples, n_classes) with probability estimates
        """
        X = self._to_dataframe(X)
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, len(self.classes_)))
        
        # Calculate log probabilities for each class
        for idx, cls in enumerate(self.classes_):
            # Start with log prior: log P(y)
            log_prior = np.log(self.class_prior_[idx])
            
            # Add log probabilities from categorical features: log P(x_cat|y)
            log_cat_prob = np.zeros(n_samples)
            for col in self.categorical_cols:
                probs = X[col].map(self.cat_prob_[col][cls]).fillna(1e-9)  # Handle unseen categories with small probability
                log_cat_prob += np.log(probs).values
            
            # Add log probabilities from Bernoulli features: log P(x_binary|y)
            # For Bernoulli: P(x|y) = p^x * (1-p)^(1-x) where p = P(x=1|y)
            log_bernoulli_prob = np.zeros(n_samples)
            for col in self.binary_cols:
                p = self.bernoulli_prob_[col][cls]
                x_vals = X[col].values
                # log P(x|y) = x*log(p) + (1-x)*log(1-p)
                log_bernoulli_prob += x_vals * np.log(p) + (1 - x_vals) * np.log(1 - p)
            
            # Add log probabilities from KDE (oldpeak): log P(oldpeak|y)
            oldpeak_vals = X[self.oldpeak_col].values.reshape(-1,1)
            log_kde = self.kde_[cls].score_samples(oldpeak_vals)
            
            # Add log probabilities from multivariate Gaussian (numerical features including age): log P(x_num|y)
            gaussian_data = X[self.numerical_gaussian_cols].values
            params = self.gaussian_params_[cls]
            log_gauss = np.array([self._log_gaussian_pdf(g, params['mu'], params['sigma_inv'], params['sigma_det']) for g in gaussian_data])
            
            # Combine all log probabilities: log P(y|X) ∝ log P(y) + log P(X|y)
            proba[:, idx] = log_prior + log_cat_prob + log_bernoulli_prob + log_kde + log_gauss
        
        # Convert log probabilities to probabilities with numerical stability (log-sum-exp trick)
        max_log_prob = proba.max(axis=1, keepdims=True)
        exp_prob = np.exp(proba - max_log_prob)
        proba_norm = exp_prob / exp_prob.sum(axis=1, keepdims=True)
        return proba_norm
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels for input samples.
        
        Parameters:
        -----------
        X : Union[pd.DataFrame, np.ndarray]
            Input features
            
        Returns:
        --------
        np.ndarray
            Array of predicted class labels
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

################################# METRICS FUNCTIONS ############################################

def compare_model_statistics(dataset: pd.DataFrame, n_rounds: int = 100):
    """
    Compare multiple classification models using repeated train-test splits.
    
    This function evaluates four models:
    1. Logistic Regression
    2. Custom Multivariate Naive Bayes
    3. Decision Trees
    4. Support Vector Machine (SVM)
    
    For each round, it:
    - Splits data into 75% train, 25% test (stratified by target)
    - Performs hyperparameter tuning using 10-fold cross-validation on training set
    - Evaluates the best model on the held-out test set using AUC-ROC
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Input dataset with features and 'num' as target column
    n_rounds : int, default=100
        Number of train-test split rounds to perform
        
    Returns:
    --------
    results : dict
        Dictionary containing AUC scores and best hyperparameters for each model
    summary_df : pd.DataFrame
        DataFrame with mean AUC (±std) and hyperparameter statistics
    """

    # Validation
    if 'num' not in dataset.columns:
        raise ValueError("Dataset must contain 'num' target column")
    if n_rounds < 1:
        raise ValueError("n_rounds must be positive")

    # Initialize results dictionary to store AUC scores and hyperparameters
    results = {
        'Logistic Regression': {'auc_scores': [], 'best_param': [], 'misclassified_indices': []},
        'Custom Naive Bayes': {'auc_scores': [], 'misclassified_indices': []},
        'Decision Trees': {'auc_scores': [], 'best_param': [], 'misclassified_indices': []},
        'SVM': {'auc_scores': [], 'best_param': [], 'misclassified_indices': []}
    }
    
    # Define hyperparameter search grids
    lr_C_values = np.linspace(1e-9, 2, 100)  # Regularization strength for Logistic Regression (smaller C = stronger regularization)
    dt_max_depth_values = list(range(1, 7))  # Maximum tree depth for Decision Trees
    svm_C_values = np.logspace(np.log10(0.0000001), np.log10(1), 100)  # Regularization for SVM (smaller C = stronger regularization)

    # Training times for each model
    training_times = {
        'Logistic Regression': [],
        'Custom Naive Bayes': [],
        'Decision Trees': [],
        'SVM': []
    }

    for _ in range(n_rounds):
        # Split into training (75%) and test (25%) sets with stratification to maintain class balance
        train_data, test_data = train_test_split(dataset, test_size=0.25, stratify=dataset['num'], shuffle=True)
        
        # ===== Prepare data for Logistic Regression =====
        # Requires one-hot encoding of categorical features and standardization
        train_data_LR = train_data.copy()
        test_data_LR = test_data.copy()
        train_data_LR = pd.get_dummies(train_data_LR, columns=list(categorical_cols.keys()), drop_first=True, dtype=int)
        test_data_LR = pd.get_dummies(test_data_LR, columns=list(categorical_cols.keys()), drop_first=True, dtype=int)
        train_data_LR, test_data_LR = data_standardization(train_data_LR, test_data_LR)

        X_train_LR = train_data_LR.drop('num', axis=1).values
        y_train_LR = train_data_LR['num'].values
        X_test_LR = test_data_LR.drop('num', axis=1).values
        y_test_LR = test_data_LR['num'].values


        # ===== Prepare data for Decision Tree =====
        # No preprocessing needed (handles categorical and numerical features natively)
        train_data_DT = train_data.copy()
        test_data_DT = test_data.copy()
        X_train_DT = train_data_DT.drop('num', axis=1).values
        y_train_DT = train_data_DT['num'].values
        X_test_DT = test_data_DT.drop('num', axis=1).values
        y_test_DT = test_data_DT['num'].values


        # ===== Prepare data for SVM =====
        # Requires one-hot encoding of categorical features and standardization
        train_data_SVM = train_data.copy()
        test_data_SVM = test_data.copy()
        train_data_SVM = pd.get_dummies(train_data_SVM, columns=list(categorical_cols.keys()), drop_first=True, dtype=int)
        test_data_SVM = pd.get_dummies(test_data_SVM, columns=list(categorical_cols.keys()), drop_first=True, dtype=int)
        train_data_SVM, test_data_SVM = data_standardization(train_data_SVM, test_data_SVM)
        X_train_SVM = train_data_SVM.drop('num', axis=1).values
        y_train_SVM = train_data_SVM['num'].values
        X_test_SVM = test_data_SVM.drop('num', axis=1).values
        y_test_SVM = test_data_SVM['num'].values


        # ===== Prepare data for Custom Naive Bayes =====
        # Requires power transformation and standardization for numerical features
        train_data_CNB = train_data.copy()
        test_data_CNB = test_data.copy()

        train_data_CNB, test_data_CNB = data_transformation(train_data_CNB, test_data_CNB)
        train_data_CNB, test_data_CNB = data_standardization(train_data_CNB, test_data_CNB)
        X_train_CNB = train_data_CNB.drop('num', axis=1).values
        y_train_CNB = train_data_CNB['num'].values
        X_test_CNB = test_data_CNB.drop('num', axis=1).values
        y_test_CNB = test_data_CNB['num'].values
        

        # ===== Logistic Regression: Hyperparameter Tuning =====
        best_lr_auc = 0
        best_lr_C = None
        # Manual grid search over regularization parameter C
        for C in lr_C_values:
            lr = LogisticRegression(C=C, max_iter=1000, solver='liblinear', random_state=42)
            cv_scores = cross_val_score(lr, X_train_LR, y_train_LR, cv=10, scoring='roc_auc')
            mean_cv_auc = np.mean(cv_scores)

            if mean_cv_auc > best_lr_auc:
                best_lr_auc = mean_cv_auc
                best_lr_C = C

        # Train final model with best hyperparameter and evaluate on test set
        start_time = time.time()
        lr_final = LogisticRegression(C=best_lr_C, max_iter=1000, solver='liblinear', random_state=42)
        lr_final.fit(X_train_LR, y_train_LR)
        end_time = time.time()
        training_times['Logistic Regression'].append(end_time - start_time)
        y_pred_proba_lr = lr_final.predict_proba(X_test_LR)[:, 1]
        lr_test_auc = roc_auc_score(y_test_LR, y_pred_proba_lr)
        
        # Track misclassified observations using optimal threshold
        optimal_threshold_lr = find_optimal_threshold(y_test_LR, y_pred_proba_lr)
        y_pred_lr = (y_pred_proba_lr >= optimal_threshold_lr).astype(int)
        misclassified_lr = test_data_LR.index[y_pred_lr != y_test_LR].tolist()
        results['Logistic Regression']['misclassified_indices'].append(misclassified_lr)

        results['Logistic Regression']['auc_scores'].append(lr_test_auc)
        results['Logistic Regression']['best_param'].append(best_lr_C)


        # ===== Decision Tree: Hyperparameter Tuning =====
        best_dt_auc = 0
        best_dt_depth = None

        # Manual grid search over max_depth parameter
        for max_depth in dt_max_depth_values:
            dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            cv_scores = cross_val_score(dt, X_train_DT, y_train_DT, cv=10, scoring='roc_auc')
            mean_cv_auc = np.mean(cv_scores)

            if mean_cv_auc > best_dt_auc:
                best_dt_auc = mean_cv_auc
                best_dt_depth = max_depth
            
        # Train final model with best hyperparameter and evaluate on test set
        start_time = time.time()
        dt_final = DecisionTreeClassifier(max_depth=best_dt_depth, random_state=42)
        dt_final.fit(X_train_DT, y_train_DT)
        end_time = time.time()
        training_times['Decision Trees'].append(end_time - start_time)
        y_pred_proba_dt = dt_final.predict_proba(X_test_DT)[:, 1]
        dt_test_auc = roc_auc_score(y_test_DT, y_pred_proba_dt)
        
        # Track misclassified observations using optimal threshold
        optimal_threshold_dt = find_optimal_threshold(y_test_DT, y_pred_proba_dt)
        y_pred_dt = (y_pred_proba_dt >= optimal_threshold_dt).astype(int)
        misclassified_dt = test_data_DT.index[y_pred_dt != y_test_DT].tolist()
        results['Decision Trees']['misclassified_indices'].append(misclassified_dt)

        results['Decision Trees']['auc_scores'].append(dt_test_auc)
        results['Decision Trees']['best_param'].append(best_dt_depth)


        # ===== SVM: Hyperparameter Tuning =====
        best_svm_auc = 0
        best_svm_C = None

        # Manual grid search over regularization parameter C (using parallel processing)
        for C in svm_C_values:
            svm = SVMClassifier(C=C, probability=True, kernel='linear', random_state=42)
            cv_scores = cross_val_score(svm, X_train_SVM, y_train_SVM, cv=10, scoring='roc_auc', n_jobs=-1)
            mean_cv_auc = np.mean(cv_scores)

            if mean_cv_auc > best_svm_auc:
                best_svm_auc = mean_cv_auc
                best_svm_C = C

        # Train final model with best hyperparameter and evaluate on test set
        start_time = time.time()
        svm_final = SVMClassifier(C=best_svm_C, probability=True, kernel='linear', random_state=42)
        svm_final.fit(X_train_SVM, y_train_SVM)
        end_time = time.time()
        training_times['SVM'].append(end_time - start_time)
        y_pred_proba_svm = svm_final.predict_proba(X_test_SVM)[:, 1]
        svm_test_auc = roc_auc_score(y_test_SVM, y_pred_proba_svm)
        
        # Track misclassified observations using optimal threshold
        optimal_threshold_svm = find_optimal_threshold(y_test_SVM, y_pred_proba_svm)
        y_pred_svm = (y_pred_proba_svm >= optimal_threshold_svm).astype(int)
        misclassified_svm = test_data_SVM.index[y_pred_svm != y_test_SVM].tolist()
        results['SVM']['misclassified_indices'].append(misclassified_svm)

        results['SVM']['auc_scores'].append(svm_test_auc)
        results['SVM']['best_param'].append(best_svm_C)


        # ===== Custom Naive Bayes =====
        # No hyperparameters to tune

        # Get feature names (excluding target column)
        feature_names = dataset.columns.tolist()
        feature_names.remove('num')

        # Initialize and train Custom Naive Bayes
        start_time = time.time()
        cnb = CustomNaiveBayesMod(
            feature_names = feature_names,
            categorical_cols=list(categorical_cols.keys()),
            binary_cols=list(binary_cols.keys()),
            numerical_gaussian_cols=list(numerical_cols.keys()),
            oldpeak_col=list(special_cols.keys())[0]
        )

        cnb.fit(X_train_CNB, y_train_CNB)
        end_time = time.time()
        training_times['Custom Naive Bayes'].append(end_time - start_time)
        y_pred_proba_cnb = cnb.predict_proba(X_test_CNB)[:, 1]
        cnb_test_auc = roc_auc_score(y_test_CNB, y_pred_proba_cnb)
        
        # Track misclassified observations using optimal threshold
        optimal_threshold_cnb = find_optimal_threshold(y_test_CNB, y_pred_proba_cnb)
        y_pred_cnb = (y_pred_proba_cnb >= optimal_threshold_cnb).astype(int)
        misclassified_cnb = test_data_CNB.index[y_pred_cnb != y_test_CNB].tolist()
        results['Custom Naive Bayes']['misclassified_indices'].append(misclassified_cnb)

        results['Custom Naive Bayes']['auc_scores'].append(cnb_test_auc)

    # ===== Compile summary statistics =====
    summary = []
    for model_name, data in results.items():
        # Calculate mean and standard deviation of AUC scores across all rounds
        mean_auc = np.mean(data['auc_scores'])
        std_auc = np.std(data['auc_scores'] / np.sqrt(n_rounds))
        
        summary.append({
            'Model': model_name,
            'Mean AUC': f"{mean_auc:.3f} ± {std_auc:.3f}",
            'Training Time (ms)': f"{np.mean(np.array(training_times[model_name])*1000):.5f} ± {(np.std(np.array(training_times[model_name])) / np.sqrt(n_rounds) * 1000):.5f}",
        })

    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('data/model_comparison_summary.csv', index=False)  
      
    return results, summary_df

################################# STATISTICAL TESTING ############################################

def paired_ttest(auc_scores_A: np.ndarray, auc_scores_B: np.ndarray) -> tuple:
    """
    Perform a paired t-test between two sets of AUC scores.
    
    This test evaluates whether there is a significant difference between the performance
    of two models by comparing their paired AUC scores from the same train-test splits.
    
    Parameters:
    -----------
    auc_scores_A : np.ndarray
        Array of AUC scores for model A
    auc_scores_B : np.ndarray
        Array of AUC scores for model B
    
    Returns:
    --------
    tuple
        - t_stat (float): t-statistic value
        - p_value (float): two-tailed p-value
        - mean_diff (float): mean difference (A - B)
        - std_diff (float): standard deviation of differences
    
    Note:
    -----
    The paired t-test assumes:
    - The differences are normally distributed (or sample size is large enough by CLT)
    - Each pair of observations comes from the same train-test split
    """
    
    # Input validation
    if len(auc_scores_A) != len(auc_scores_B):
        raise ValueError("Both arrays must have the same length")
    
    # Calculate differences
    differences = auc_scores_A - auc_scores_B
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)
    
    # Calculate t-statistic
    t_stat = mean_diff / (std_diff / np.sqrt(n))
    
    # Calculate two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-1))
    
    return t_stat, p_value, mean_diff, std_diff


def pairwise_model_comparison(results: dict, alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform pairwise paired t-tests between all models with Bonferroni correction.
    
    This function compares each pair of models using paired t-tests on their AUC scores,
    and applies Bonferroni correction to control the family-wise error rate when
    performing multiple comparisons.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results with 'auc_scores' for each model
        (output from compare_model_statistics function)
    alpha : float, default=0.05
        Significance level before correction
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - Model A: First model name
        - Model B: Second model name
        - Mean Diff (A-B): Mean difference in AUC scores
        - Std Diff: Standard deviation of differences
        - t-statistic: Calculated t-statistic
        - p-value: Two-tailed p-value
        - Significant (Bonferroni): Whether difference is significant after correction
        - Bonferroni α: Corrected significance level
    
    Note:
    -----
    Bonferroni correction: α_corrected = α / n_comparisons
    For k models, there are k(k-1)/2 pairwise comparisons
    """
    # Get model names and their AUC scores
    model_names = list(results.keys())
    n_models = len(model_names)
    
    # Calculate number of pairwise comparisons
    n_comparisons = n_models * (n_models - 1) // 2
    
    # Apply Bonferroni correction
    alpha_corrected = alpha / n_comparisons
    
    # Store results for all pairwise comparisons
    comparison_results = []
    
    # Perform pairwise comparisons
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model_A = model_names[i]
            model_B = model_names[j]
            
            # Get AUC scores for both models
            auc_scores_A = np.array(results[model_A]['auc_scores'])
            auc_scores_B = np.array(results[model_B]['auc_scores'])
            
            # Perform paired t-test
            t_stat, p_value, mean_diff, std_diff = paired_ttest(auc_scores_A, auc_scores_B)
            
            # Determine if significant after Bonferroni correction
            is_significant = p_value < alpha_corrected
            
            comparison_results.append({
                'Model A': model_A,
                'Model B': model_B,
                'Mean Diff (A-B)': f"{mean_diff:.4f}",
                'Std Diff': f"{std_diff:.4f}",
                't-statistic': f"{t_stat:.4f}",
                'p-value': f"{p_value:.6f}",
                'Significant (Bonferroni)': 'Yes' if is_significant else 'No',
                'Bonferroni α': f"{alpha_corrected:.6f}"
            })
    
    # Create DataFrame and save to CSV
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv('data/pairwise_ttest_results.csv', index=False)
    
    return comparison_df


def print_statistical_summary(comparison_df: pd.DataFrame):
    """
    Print a formatted summary of pairwise statistical comparisons.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame output from pairwise_model_comparison function
    """
    print("\n" + "="*80)
    print("PAIRWISE STATISTICAL COMPARISON OF MODELS (Paired t-test with Bonferroni)")
    print("="*80 + "\n")
    
    for _, row in comparison_df.iterrows():
        print(f"{row['Model A']} vs {row['Model B']}:")
        print(f"  Mean Difference: {row['Mean Diff (A-B)']} ± {row['Std Diff']}")
        print(f"  t-statistic: {row['t-statistic']}")
        print(f"  p-value: {row['p-value']}")
        print(f"  Significant (α = {row['Bonferroni α']}): {row['Significant (Bonferroni)']}")
        print()
    
    print("="*80)
    
    # Summary of significant differences
    significant_comparisons = comparison_df[comparison_df['Significant (Bonferroni)'] == 'Yes']
    if len(significant_comparisons) > 0:
        print(f"\nSignificant differences found in {len(significant_comparisons)} comparison(s):")
        for _, row in significant_comparisons.iterrows():
            print(f"  • {row['Model A']} vs {row['Model B']}")
    else:
        print("\nNo significant differences found between any models after Bonferroni correction.")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    pass
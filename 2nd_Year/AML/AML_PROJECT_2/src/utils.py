import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from scipy.linalg import null_space

def sigmoid_kernel(X, Y=None, gamma=None, coef0=None):
    """
    Compute the sigmoid kernel matrix for input data.
    K(x, y) = tanh(gamma * <x, y> + coef0)

    Parameters:
        X (ndarray): First data matrix.
        Y (ndarray, optional): Second data matrix. If None, uses X.
        gamma (float, optional): Kernel gamma parameter.
        coef0 (float, optional): Kernel coef0 parameter.

    Returns:
        K (ndarray): Kernel matrix.
    """
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 0.01
    if coef0 is None:
        coef0 = -1
    K = np.tanh(gamma * np.dot(X, Y.T) + coef0)
    return K

def sigmoid_kernel_normalized(X, Y=None, gamma=None, coef0=None):
    """
    Compute the sigmoid kernel matrix using normalized dot product.
    K(x, y) = tanh(gamma * (<x, y> / (||x|| * ||y||)) + coef0)
    
    The normalized dot product is equivalent to the cosine similarity.

    Parameters:
        X (ndarray): First data matrix.
        Y (ndarray, optional): Second data matrix. If None, uses X.
        gamma (float, optional): Kernel gamma parameter.
        coef0 (float, optional): Kernel coef0 parameter.

    Returns:
        K (ndarray): Kernel matrix.
    """
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0  # Different default since normalized dot product is in [-1, 1]
    if coef0 is None:
        coef0 = 0.0
    
    # Compute norms
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    
    # Avoid division by zero
    X_norm = np.maximum(X_norm, 1e-10)
    Y_norm = np.maximum(Y_norm, 1e-10)
    
    # Normalized dot product (cosine similarity)
    cos_sim = np.dot(X, Y.T) / (X_norm @ Y_norm.T)
    
    K = np.tanh(gamma * cos_sim + coef0)
    return K

def scan_param_grid(X, gammas, coef0s, tol=1e-10):
    """
    Scan a grid of gamma and coef0 values, computing kernel indefiniteness measures.

    Parameters:
        X (ndarray): Data matrix.
        gammas (array-like): List of gamma values.
        coef0s (array-like): List of coef0 values.
        tol (float): Tolerance for eigenvalue negativity.

    Returns:
        H1 (ndarray): Matrix of E_minus values.
        H2 (ndarray): Matrix of f_neg values.
        H3 (ndarray): Matrix of Nneg values.
        H4 (ndarray): Matrix of CPD status (True if CPD, False otherwise).
    """
    n_c0 = len(coef0s)
    n_g = len(gammas)
    H1 = np.zeros((n_c0, n_g))  # E_minus
    H2 = np.zeros((n_c0, n_g))  # f_neg
    H3 = np.zeros((n_c0, n_g))  # Nneg
    H4 = np.zeros((n_c0, n_g), dtype=bool)  # CPD status

    for i, c0 in enumerate(coef0s):
        for j, g in enumerate(gammas):
            K = sigmoid_kernel(X, gamma=g, coef0=c0)
            m = indefiniteness_measures(K, tol=tol)
            H1[i, j] = m["E_minus"]
            H2[i, j] = m["f_neg"]
            H3[i, j] = m["Nneg"]
            H4[i, j] = is_cpd(K, tol=tol)
    return H1, H2, H3, H4

def indefiniteness_measures(K, tol=1e-10):
    """
    Compute several measures of indefiniteness from the eigenvalues of a kernel matrix.

    Parameters:
        K (ndarray): Kernel matrix.
        tol (float): Tolerance for considering eigenvalues negative.

    Returns:
        dict: Dictionary of measures (lam_min, lam_max_abs, Nneg, f_neg, I_indef, E_minus, r).
    """
    evals = np.linalg.eigvalsh(K)
    lam_min = np.min(evals)
    lam_max_abs = np.max(np.abs(evals))
    neg_mask = evals < -tol
    pos_mask = evals > tol
    Nneg = int(np.sum(neg_mask))
    n = K.shape[0]
    pos_sum = np.sum(evals[pos_mask])
    neg_sum_abs = np.sum(np.abs(evals[neg_mask]))
    total_abs_sum = np.sum(np.abs(evals))

    I_indef = np.nan
    if pos_sum > 0:
        I_indef = neg_sum_abs / pos_sum

    E_minus = 0.0
    if total_abs_sum > 0:
        E_minus = neg_sum_abs / total_abs_sum

    f_neg = Nneg / n

    r = 0.0
    if lam_max_abs > 0:
        r = np.abs(lam_min) / lam_max_abs

    return {
        "lam_min": lam_min,
        "lam_max_abs": lam_max_abs,
        "Nneg": Nneg,
        "f_neg": f_neg,
        "I_indef": I_indef,
        "E_minus": E_minus,
        "r": r,
    }

def shift_to_psd(K, eps=0.0):
    """
    Shift kernel matrix to positive semi-definite (PSD) by adding c*I, where c >= -lambda_min + eps.

    Parameters:
        K (ndarray): Kernel matrix.
        eps (float): Small positive value to ensure strict PSD.

    Returns:
        K_shift (ndarray): Shifted PSD kernel matrix.
    """
    w, _ = np.linalg.eigh(K)
    lam_min = w.min()
    c = max(0.0, -lam_min + eps)
    return K + c * np.eye(K.shape[0])

def clip_to_psd(K):
    """
    Project kernel matrix onto PSD cone by setting negative eigenvalues to zero.

    Parameters:
        K (ndarray): Kernel matrix.

    Returns:
        K_clip (ndarray): PSD kernel matrix with negative eigenvalues clipped.
    """
    w, U = np.linalg.eigh(K)
    w_clipped = np.maximum(w, 0.0)
    K_clip = (U * w_clipped) @ U.T
    return K_clip

def clipnorm_nearest_psd(A, max_iter=100, tol=1e-10):
    """
    Compute the nearest PSD matrix to A using clipnorm's algorithm.

    Parameters:
        A (ndarray): Input matrix.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        K (ndarray): Nearest PSD matrix.
    """
    Y = A.copy()
    delta_S = np.zeros_like(A)
    for k in range(max_iter):
        R = Y - delta_S
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals_clipped = np.maximum(eigvals, 0)
        X = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        delta_S = X - R
        if np.linalg.norm(X - Y, 'fro') < tol:
            Y = X
            break
        Y = X
    K = (Y + Y.T) / 2
    diag = np.sqrt(np.outer(np.diag(K), np.diag(K)))
    K = K / diag
    return K

def find_best_C(K_train_list, K_val_list, y_train_list, y_val_list, C_values):
    """
    Find the best C value for SVM using k-fold cross-validation.

    Parameters:
        K_train_list (list): List of training kernel matrices for each fold.
        K_val_list (list): List of validation kernel matrices for each fold.
        y_train_list (list): List of training labels for each fold.
        y_val_list (list): List of validation labels for each fold.
        C_values (array-like): List of C values to test.

    Returns:
        best_C (float): Best C value found.
        best_cv_f1 (float): Best mean cross-validation F1 score.
    """
    best_C = None
    best_cv_f1 = -np.inf
    for C in C_values:
        fold_scores = []
        for i in range(len(K_train_list)):
            svm = SVC(kernel='precomputed', C=C, random_state=42)
            svm.fit(K_train_list[i], y_train_list[i])
            y_pred = svm.predict(K_val_list[i])
            f1 = f1_score(y_val_list[i], y_pred)
            fold_scores.append(f1)
        mean_cv_f1 = np.mean(fold_scores)
        if mean_cv_f1 > best_cv_f1:
            best_cv_f1 = mean_cv_f1
            best_C = C
    return best_C, best_cv_f1

def process_parameter_combination(gamma, c, X_train_std, y_train, X_test_std, y_test, C_values):
    """
    Process one (gamma, coef0) combination:
    - Compute kernels (original + 3 corrections)
    - Find best C for each method using k-fold CV
    - Evaluate on test set

    Parameters:
        gamma (float): Kernel gamma parameter.
        c (float): Kernel coef0 parameter.
        X_train_std (ndarray): Standardized training data.
        y_train (ndarray): Training labels.
        X_test_std (ndarray): Standardized test data.
        y_test (ndarray): Test labels.
        C_values (array-like): List of C values to test.

    Returns:
        result (dict): Dictionary with results for this combination.
    """
    result = {
        'gamma': gamma,
        'coef0': c,
        'original': {},
        'shift': {},
        'clip': {},
        'clipnorm': {}
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    K_train_original = []
    K_val_original = []
    K_train_shift = []
    K_val_shift = []
    K_train_clip = []
    K_val_clip = []
    K_train_clipnorm = []
    K_val_clipnorm = []
    y_train_folds = []
    y_val_folds = []

    # Compute all kernel matrices for k-fold
    for train_idx, val_idx in skf.split(X_train_std, y_train):
        X_fold_train = X_train_std[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train_std[val_idx]
        y_fold_val = y_train[val_idx]

        # Original non-PSD kernel
        K_tr_orig = sigmoid_kernel(X_fold_train, X_fold_train, gamma, c)
        K_vl_orig = sigmoid_kernel(X_fold_val, X_fold_train, gamma, c)

        # Apply corrections to training kernel
        K_tr_shift = shift_to_psd(K_tr_orig)
        K_tr_clip = clip_to_psd(K_tr_orig)
        K_tr_clipnorm = clipnorm_nearest_psd(K_tr_orig)

        # Store kernels
        K_train_original.append(K_tr_orig)
        K_val_original.append(K_vl_orig)
        K_train_shift.append(K_tr_shift)
        K_val_shift.append(K_vl_orig)
        K_train_clip.append(K_tr_clip)
        K_val_clip.append(K_vl_orig)
        K_train_clipnorm.append(K_tr_clipnorm)
        K_val_clipnorm.append(K_vl_orig)

        y_train_folds.append(y_fold_train)
        y_val_folds.append(y_fold_val)

    # Find best C for each method
    best_C_orig, cv_f1_orig = find_best_C(K_train_original, K_val_original, 
                                          y_train_folds, y_val_folds, C_values)
    best_C_shift, cv_f1_shift = find_best_C(K_train_shift, K_val_shift, 
                                             y_train_folds, y_val_folds, C_values)
    best_C_clip, cv_f1_clip = find_best_C(K_train_clip, K_val_clip, 
                                           y_train_folds, y_val_folds, C_values)
    best_C_clipnorm, cv_f1_clipnorm = find_best_C(K_train_clipnorm, K_val_clipnorm, 
                                               y_train_folds, y_val_folds, C_values)

    # Train on full training set and evaluate on test set
    K_train_full_orig = sigmoid_kernel(X_train_std, X_train_std, gamma, c)
    K_test_orig = sigmoid_kernel(X_test_std, X_train_std, gamma, c)

    # Original
    svm_orig = SVC(kernel='precomputed', C=best_C_orig, random_state=42)
    svm_orig.fit(K_train_full_orig, y_train)
    f1_orig = f1_score(y_test, svm_orig.predict(K_test_orig))
    result['original']['f1'] = f1_orig
    result['original']['best_C'] = best_C_orig
    result['original']['cv_f1'] = cv_f1_orig

    # Shift
    K_train_full_shift = shift_to_psd(K_train_full_orig)
    svm_shift = SVC(kernel='precomputed', C=best_C_shift, random_state=42)
    svm_shift.fit(K_train_full_shift, y_train)
    f1_shift = f1_score(y_test, svm_shift.predict(K_test_orig))
    result['shift']['f1'] = f1_shift
    result['shift']['best_C'] = best_C_shift
    result['shift']['cv_f1'] = cv_f1_shift

    # Clip
    K_train_full_clip = clip_to_psd(K_train_full_orig)
    svm_clip = SVC(kernel='precomputed', C=best_C_clip, random_state=42)
    svm_clip.fit(K_train_full_clip, y_train)
    f1_clip = f1_score(y_test, svm_clip.predict(K_test_orig))
    result['clip']['f1'] = f1_clip
    result['clip']['best_C'] = best_C_clip
    result['clip']['cv_f1'] = cv_f1_clip

    # clipnorm
    K_train_full_clipnorm = clipnorm_nearest_psd(K_train_full_orig)
    svm_clipnorm = SVC(kernel='precomputed', C=best_C_clipnorm, random_state=42)
    svm_clipnorm.fit(K_train_full_clipnorm, y_train)
    f1_clipnorm = f1_score(y_test, svm_clipnorm.predict(K_test_orig))
    result['clipnorm']['f1'] = f1_clipnorm
    result['clipnorm']['best_C'] = best_C_clipnorm
    result['clipnorm']['cv_f1'] = cv_f1_clipnorm

    return result

def scan_param_grid_normalized(X, gammas, coef0s, tol=1e-10):
    """
    Scan a grid of gamma and coef0 values using normalized sigmoid kernel,
    computing kernel indefiniteness measures.

    Parameters:
        X (ndarray): Data matrix.
        gammas (array-like): List of gamma values.
        coef0s (array-like): List of coef0 values.
        tol (float): Tolerance for eigenvalue negativity.

    Returns:
        H1 (ndarray): Matrix of E_minus values.
        H2 (ndarray): Matrix of f_neg values.
        H3 (ndarray): Matrix of Nneg values.
        H4 (ndarray): Matrix of CPD status (True if CPD, False otherwise).
    """
    n_c0 = len(coef0s)
    n_g = len(gammas)
    H1 = np.zeros((n_c0, n_g))  # E_minus
    H2 = np.zeros((n_c0, n_g))  # f_neg
    H3 = np.zeros((n_c0, n_g))  # Nneg
    H4 = np.zeros((n_c0, n_g), dtype=bool)  # CPD status

    for i, c0 in enumerate(coef0s):
        for j, g in enumerate(gammas):
            K = sigmoid_kernel_normalized(X, gamma=g, coef0=c0)
            m = indefiniteness_measures(K, tol=tol)
            H1[i, j] = m["E_minus"]
            H2[i, j] = m["f_neg"]
            H3[i, j] = m["Nneg"]
            H4[i, j] = is_cpd(K, tol=tol)
    return H1, H2, H3, H4

def process_parameter_combination_normalized(gamma, c, X_train_std, y_train, X_test_std, y_test, C_values):
    """
    Process one (gamma, coef0) combination using normalized sigmoid kernel:
    - Compute kernels (original + 3 corrections)
    - Find best C for each method using k-fold CV
    - Evaluate on test set

    Parameters:
        gamma (float): Kernel gamma parameter.
        c (float): Kernel coef0 parameter.
        X_train_std (ndarray): Standardized training data.
        y_train (ndarray): Training labels.
        X_test_std (ndarray): Standardized test data.
        y_test (ndarray): Test labels.
        C_values (array-like): List of C values to test.

    Returns:
        result (dict): Dictionary with results for this combination.
    """
    result = {
        'gamma': gamma,
        'coef0': c,
        'original': {},
        'shift': {},
        'clip': {},
        'clipnorm': {}
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    K_train_original = []
    K_val_original = []
    K_train_shift = []
    K_val_shift = []
    K_train_clip = []
    K_val_clip = []
    K_train_clipnorm = []
    K_val_clipnorm = []
    y_train_folds = []
    y_val_folds = []

    # Compute all kernel matrices for k-fold
    for train_idx, val_idx in skf.split(X_train_std, y_train):
        X_fold_train = X_train_std[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train_std[val_idx]
        y_fold_val = y_train[val_idx]

        # Original non-PSD kernel (normalized)
        K_tr_orig = sigmoid_kernel_normalized(X_fold_train, X_fold_train, gamma, c)
        K_vl_orig = sigmoid_kernel_normalized(X_fold_val, X_fold_train, gamma, c)

        # Apply corrections to training kernel
        K_tr_shift = shift_to_psd(K_tr_orig)
        K_tr_clip = clip_to_psd(K_tr_orig)
        K_tr_clipnorm = clipnorm_nearest_psd(K_tr_orig)

        # Store kernels
        K_train_original.append(K_tr_orig)
        K_val_original.append(K_vl_orig)
        K_train_shift.append(K_tr_shift)
        K_val_shift.append(K_vl_orig)
        K_train_clip.append(K_tr_clip)
        K_val_clip.append(K_vl_orig)
        K_train_clipnorm.append(K_tr_clipnorm)
        K_val_clipnorm.append(K_vl_orig)

        y_train_folds.append(y_fold_train)
        y_val_folds.append(y_fold_val)

    # Find best C for each method
    best_C_orig, cv_f1_orig = find_best_C(K_train_original, K_val_original, 
                                          y_train_folds, y_val_folds, C_values)
    best_C_shift, cv_f1_shift = find_best_C(K_train_shift, K_val_shift, 
                                             y_train_folds, y_val_folds, C_values)
    best_C_clip, cv_f1_clip = find_best_C(K_train_clip, K_val_clip, 
                                           y_train_folds, y_val_folds, C_values)
    best_C_clipnorm, cv_f1_clipnorm = find_best_C(K_train_clipnorm, K_val_clipnorm, 
                                               y_train_folds, y_val_folds, C_values)

    # Train on full training set and evaluate on test set
    K_train_full_orig = sigmoid_kernel_normalized(X_train_std, X_train_std, gamma, c)
    K_test_orig = sigmoid_kernel_normalized(X_test_std, X_train_std, gamma, c)

    # Original
    svm_orig = SVC(kernel='precomputed', C=best_C_orig, random_state=42)
    svm_orig.fit(K_train_full_orig, y_train)
    f1_orig = f1_score(y_test, svm_orig.predict(K_test_orig))
    result['original']['f1'] = f1_orig
    result['original']['best_C'] = best_C_orig
    result['original']['cv_f1'] = cv_f1_orig

    # Shift
    K_train_full_shift = shift_to_psd(K_train_full_orig)
    svm_shift = SVC(kernel='precomputed', C=best_C_shift, random_state=42)
    svm_shift.fit(K_train_full_shift, y_train)
    f1_shift = f1_score(y_test, svm_shift.predict(K_test_orig))
    result['shift']['f1'] = f1_shift
    result['shift']['best_C'] = best_C_shift
    result['shift']['cv_f1'] = cv_f1_shift

    # Clip
    K_train_full_clip = clip_to_psd(K_train_full_orig)
    svm_clip = SVC(kernel='precomputed', C=best_C_clip, random_state=42)
    svm_clip.fit(K_train_full_clip, y_train)
    f1_clip = f1_score(y_test, svm_clip.predict(K_test_orig))
    result['clip']['f1'] = f1_clip
    result['clip']['best_C'] = best_C_clip
    result['clip']['cv_f1'] = cv_f1_clip

    # clipnorm
    K_train_full_clipnorm = clipnorm_nearest_psd(K_train_full_orig)
    svm_clipnorm = SVC(kernel='precomputed', C=best_C_clipnorm, random_state=42)
    svm_clipnorm.fit(K_train_full_clipnorm, y_train)
    f1_clipnorm = f1_score(y_test, svm_clipnorm.predict(K_test_orig))
    result['clipnorm']['f1'] = f1_clipnorm
    result['clipnorm']['best_C'] = best_C_clipnorm
    result['clipnorm']['cv_f1'] = cv_f1_clipnorm

    return result

def is_cpd(K, tol=1e-9):
    """
    Returns True if matrix K is Conditionally Positive Definite (CPD).
    
    Parameters:
    - K: The kernel matrix (n x n)
    - tol: Numerical tolerance (default 1e-9) to handle floating point noise.
           Using 0.0 is dangerous; use a small epsilon.
    """
    n = K.shape[0]
    
    # 1. Get basis vectors for the subspace where sum(v) = 0
    # null_space uses SVD to find an orthonormal basis for the null space of e^T
    e = np.ones((n, 1))
    V = null_space(e.T) 
    
    # 2. Project K onto this subspace (reduces dim to n-1)
    # The SVM constraint sum(alpha * y) = 0 effectively operates here.
    K_projected = V.T @ K @ V
    
    # 3. Check the smallest eigenvalue of the projected matrix
    # We use eigvalsh because K_projected is guaranteed symmetric
    min_eig = np.linalg.eigvalsh(K_projected).min()
    
    # Returns True only if strictly positive (CPD), false otherwise.
    return min_eig > tol

def main():
    """
    Main function placeholder for consistency.
    """
    pass

if __name__ == "__main__":
    main()

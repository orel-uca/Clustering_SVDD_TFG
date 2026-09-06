
import os
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt

from models import *
from svdd_class import *
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist
from sklearn import metrics as sk_metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

def Kernel(a, b, kernel="linear", sigma=1, degree=2):
    """
    Compute the kernel matrix between two sets of vectors.
    This function calculates various kernel transformations between input matrices,
    commonly used in machine learning algorithms like Support Vector Machines.
    Parameters
    ----------
        a : array-like, shape (n_samples_a, n_features)
            First input matrix where each row represents a sample.
        b : array-like, shape (n_samples_b, n_features)
            Second input matrix where each row represents a sample.
        kernel : str, optional (default="linear")
            Type of kernel to compute. Options are:
            - "linear": Linear kernel (dot product)
            - "rbf": Radial Basis Function (Gaussian) kernel
            - "polynomial": Polynomial kernel
        sigma : float, optional (default=1)
            Bandwidth parameter for the RBF kernel. Controls the influence radius
            of samples. Only used when kernel="rbf".
        degree : int, optional (default=2)
            Degree of the polynomial kernel. Only used when kernel="polynomial".
    Returns
    -------
        ndarray, shape (n_samples_a, n_samples_b)
        Kernel matrix representing the similarity between samples in a and b.
    """
    
    if kernel == "linear":
        return np.dot(a, b.T)
    elif kernel == "rbf":
        sq_dists = cdist(a, b, 'sqeuclidean')
        # return np.exp(-sq_dists / sigma)
        return np.exp(-sq_dists / (2* sigma**2))
    elif kernel == "polynomial":
        return (1 + np.dot(a, b.T)) ** degree
    else:
        raise ValueError(f"Kernel '{kernel}' no reconocido. Los valores válidos son 'linear', 'rbf', 'polynomial'.")

def predict_scores(sol, new_points):
    """
    Predicts the scores of new points with respect to solution object.
    Parameters:
    -----------
        sol (Solution-like object): An object containing the solution parameters, which includes:
            - x: Training data points (size: n, d).
            - alpha: Coefficients for each cluster (size: n).
            - RR: Radius of the sphere.
            - c: Centers of the clusters (size: d).
            - kernel: Kernel type used for calculations.
            - sigma: Parameter for the kernel (if applicable).
            - degree: Degree of the polynomial kernel (if applicable).
        new_points: Array of new data points to predict scores for (size: m, d).
    Returns:
    --------
        scores: score for each new point with respect to the clusters (size: m,).
    """
    
    x = sol.x
    
    alpha = sol.alpha  # Vector (n,)
    RR = sol.R         # Scalar
    c = sol.c          # Vector (d,) or None
    kernel = sol.kernel
    sigma = sol.sigma
    degree = sol.degree

    if c is None: # Si los centros son None, es que se ha usado kernels (Dual SVDD)
        # Kernel entre puntos de entrenamiento
        K_x_x = Kernel(x, x, kernel=kernel, sigma=sigma, degree=degree)  # (n, n)
        
        # Término constante del radio al cuadrado
        C_val = alpha.T @ K_x_x @ alpha # Scalar

        # Kernel entre los nuevos puntos y los de entrenamiento
        K_pt_x = Kernel(new_points, x, kernel=kernel, sigma=sigma, degree=degree)  # (m, n)

        # Término del kernel de cada punto consigo mismo (diagonal)
        # Para 'rbf', es 1. Para 'linear', es np.sum(new_points**2, axis=1)
        if kernel == 'rbf':
            K_pt_pt = np.ones(len(new_points))  # (m,)
        else: # 'linear' or 'polynomial'
            K_pt_pt = np.diagonal(Kernel(new_points, new_points, kernel=kernel, degree=degree))
            # K_pt_pt = K_pt_pt = np.diagonal(Kernel(new_points, new_points, kernel=kernel, degree=degree))

        # Calcular los scores
        scores = K_pt_pt - 2 * (K_pt_x @ alpha).T + C_val - RR # (m,)

    else: # Primal SVDD
        # Distancia al cuadrado de cada punto al centro c
        scores = cdist(new_points, c.reshape(1, -1))[:, 0]**2 - RR

    # Devuelve el score, el score (de nuevo) y el índice del cluster (siempre 0)
    return scores.T

def predict_scores_multisphere(sol, new_points):
    """
    Computes anomaly scores for a multi-sphere SVDD solution.
    Each point is assigned to the sphere that gives the smallest SVDD score.
    
    Params:
    -------
        sol (MultiSphereSolution-like object): Solution object containing the list of fitted SVDD components.
        new_points (np.ndarray): Points to score, with one sample per row.
    
    Returns:
    --------
        tuple: Tuple (scores, labels, scores_all), where scores are the minimum scores, labels are the closest component labels, and scores_all contains all component scores.
    """
    scores_all = []

    for component in sol.components:
        scores_j = predict_scores(component, new_points).reshape(-1)
        scores_all.append(scores_j)

    scores_all = np.vstack(scores_all).T 

    component_pos = np.argmin(scores_all, axis=1)
    scores = np.min(scores_all, axis=1)
    
    labels = np.array([sol.components[pos].cluster_id for pos in component_pos])

    return scores, labels, scores_all

def predict_scores_any(sol, new_points):
    """
    Computes anomaly scores using any supported fitted model.
    The function dispatches automatically according to the attributes stored in the solution object.
    
    Params:
    -------
        sol (object): Fitted solution object from SVDD, multi-sphere SVDD, OCSVM, or SVM.
        new_points (np.ndarray): Points to score, with one sample per row.
    
    Returns:
    --------
        np.ndarray: Score for each point, using the sign convention of the corresponding model solution.
    """
    model_type = getattr(sol, "model_type", None)

    if model_type == "ocsvm":
        return -sol.model.decision_function(new_points).reshape(-1)

    elif model_type == "svm":
        return sol.model.decision_function(new_points).reshape(-1)

    elif hasattr(sol, "components"):
        return predict_scores_multisphere(sol, new_points)[0]

    else:
        return predict_scores(sol, new_points)
def calculate_single_metric(metric_name, scores, y_pred, y_true, val_idx, test_idx,
                                current_max_val_score, current_test_score_at_max_val, current_sigma_at_max_val,
                                candidate_sigma, use_labels_for_roc_pr=False):
    """
    Calculates a single performance metric on validation and test sets.
    If the validation score improves, it updates the best validation score,
    the corresponding test score, and the sigma value.

    Params:
    -------
        metric_name (str): The name of the metric to calculate.
        Supported: 'roc', 'pr', 'f1', 'mcc', 'bal', 'acc'.
        scores (np.array): Raw scores from the model for all data points.
        y_pred (np.array): Predicted labels (0 or 1) for all data points.
        y_true (np.array): True labels (0 or 1) for all data points.
        val_idx (np.array): Indices for the validation set.
        test_idx (np.array): Indices for the test set.
        current_max_val_score (float): The best validation score found so far for this metric.
        current_test_score_at_max_val (float): The test score corresponding to current_max_val_score.
        current_sigma_at_max_val (float): The sigma value corresponding to current_max_val_score.
        candidate_sigma (float): The current sigma value being evaluated.
        use_labels_for_roc_pr (bool): If True, use y_pred for ROC/PR AUC calculation; otherwise, use scores.

    Returns:
    --------
        tuple: (updated_max_val_score, updated_test_score_at_max_val, updated_sigma_at_max_val)
    """
        
    val_metric_value = 0.0
    test_metric_value = 0.0

    # Initialize return values to current bests
    updated_max_val_score = current_max_val_score
    updated_test_score_at_max_val = current_test_score_at_max_val
    updated_sigma_at_max_val = current_sigma_at_max_val

    if metric_name == 'roc':
        val_score_source = scores[val_idx] if not use_labels_for_roc_pr else y_pred[val_idx]
        test_score_source = scores[test_idx] if not use_labels_for_roc_pr else y_pred[test_idx]
        
        fpr_val, tpr_val, _ = sk_metrics.roc_curve(y_true[val_idx], val_score_source, pos_label=1)
        val_metric_value = sk_metrics.auc(fpr_val, tpr_val)
        
        if val_metric_value >= current_max_val_score:
            fpr_test, tpr_test, _ = sk_metrics.roc_curve(y_true[test_idx], test_score_source, pos_label=1)
            test_metric_value = sk_metrics.auc(fpr_test, tpr_test)
            
            updated_max_val_score = val_metric_value
            updated_test_score_at_max_val = test_metric_value
            updated_sigma_at_max_val = candidate_sigma
            
    elif metric_name == 'pr':
        val_score_source = scores[val_idx] if not use_labels_for_roc_pr else y_pred[val_idx]
        test_score_source = scores[test_idx] if not use_labels_for_roc_pr else y_pred[test_idx]

        val_metric_value = sk_metrics.average_precision_score(y_true[val_idx], val_score_source, pos_label=1)
        
        if val_metric_value >= current_max_val_score:
            test_metric_value = sk_metrics.average_precision_score(y_true[test_idx], test_score_source, pos_label=1)
            
            updated_max_val_score = val_metric_value
            updated_test_score_at_max_val = test_metric_value
            updated_sigma_at_max_val = candidate_sigma

    elif metric_name == 'f1':
        val_metric_value = sk_metrics.f1_score(y_true[val_idx], y_pred[val_idx], zero_division=0)
        
        if val_metric_value >= current_max_val_score:
            test_metric_value = sk_metrics.f1_score(y_true[test_idx], y_pred[test_idx], zero_division=0)
            
            updated_max_val_score = val_metric_value
            updated_test_score_at_max_val = test_metric_value
            updated_sigma_at_max_val = candidate_sigma
            
    elif metric_name == 'mcc':
        val_metric_value = sk_metrics.matthews_corrcoef(y_true[val_idx], y_pred[val_idx])
        
        if val_metric_value >= current_max_val_score:
            test_metric_value = sk_metrics.matthews_corrcoef(y_true[test_idx], y_pred[test_idx])
            
            updated_max_val_score = val_metric_value
            updated_test_score_at_max_val = test_metric_value
            updated_sigma_at_max_val = candidate_sigma
            
    elif metric_name == 'bal':
        val_metric_value = sk_metrics.balanced_accuracy_score(y_true[val_idx], y_pred[val_idx])
        
        if val_metric_value >= current_max_val_score:
            test_metric_value = sk_metrics.balanced_accuracy_score(y_true[test_idx], y_pred[test_idx])
            
            updated_max_val_score = val_metric_value
            updated_test_score_at_max_val = test_metric_value
            updated_sigma_at_max_val = candidate_sigma

    elif metric_name == 'acc':
        val_metric_value = sk_metrics.accuracy_score(y_true[val_idx], y_pred[val_idx])
        
        if val_metric_value >= current_max_val_score:
            test_metric_value = sk_metrics.accuracy_score(y_true[test_idx], y_pred[test_idx])
            
            updated_max_val_score = val_metric_value
            updated_test_score_at_max_val = test_metric_value
            updated_sigma_at_max_val = candidate_sigma
    else:
        # Optionally, raise an error or print a warning for unsupported metric_name
        print(f"Warning: Metric '{metric_name}' is not supported.")
        pass

    return updated_max_val_score, updated_test_score_at_max_val, updated_sigma_at_max_val


def set_global_plot_limits(ax, data, padding=0.15):
    """
    Function for adjusting plot limits.
    
    Parameters:
    -----------
        ax (matplotlib.axes.Axes): Axis where the limits will be applied.
        data (np.ndarray): Two-dimensional data array with shape (2, n_samples).
        padding (float) Controls the additional margin around the data.
    """

    x_min, x_max = data[0, :].min(), data[0, :].max()
    y_min, y_max = data[1, :].min(), data[1, :].max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Evitar problemas si todos los puntos tienen la misma coordenada
    if x_range == 0:
        x_range = 1.0
    if y_range == 0:
        y_range = 1.0

    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax.set_aspect("equal", adjustable="box")


def draw_primal_spheres(ax, sol):
    """
    Draws primal SVDD spheres on a matplotlib axis.
    It plots each available center and its corresponding radius.
    
    Params:
    -------
        ax (matplotlib.axes.Axes): Axis where the centers and circles will be drawn.
        sol (Solution-like object): Single-sphere or multi-sphere SVDD solution.

    """

    components = sol.components if hasattr(sol, "components") else [sol]

    for j, component in enumerate(components):
        if component.R is None or component.R <= 0:
            continue

        c = component.c
        R = np.sqrt(component.R)

        ax.plot(c[0], c[1], marker="*", color="cyan", markersize=6, markeredgecolor="black", markeredgewidth=0.2)

        circle = Circle((c[0], c[1]), R, color="blue", fill=False, linewidth=1.5, alpha=0.7)

        ax.add_patch(circle)

def DrawCurves(sol, anom_frac, method="svdd", ax=None, show_plot_details=True, colorbar_shown=True):
    """
    Visualizes the decision boundaries and classification results of a model based on the provided solution object drom dual MSVDD model version.
    Parameters:
    -----------
        sol (Solution-like object): Fitted solution used to compute decision scores.
        anom_frac (float): Fraction of anomalies used in the experiment.
        method (str): Name of the method shown in titles and saved filenames.
        ax (matplotlib.axes.Axes or None): Axis where the plot is drawn. If None, a new figure is created.
        show_plot_details (bool): Whether to display extra plot details such as titles and legends.
        colorbar_shown (bool): Whether to draw a colorbar for the score background.
    """
    
    
    ### Leer variables de sol:
    if hasattr(sol, "components"):
        x= sol.data[:sol.ntrain, :]
    else:
        x= sol.x
    Nu = sol.Nu

    m = 100  # Aumentar resolución para mejor visualización

    # CREAR LA MALLA BASÁNDOSE EN LOS DATOS REALES
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, m), np.linspace(y_min, y_max, m))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    scores_grid = predict_scores_any(sol, grid)
    scores_grid = np.asarray(scores_grid).reshape(xx.shape)
    
    # Clasificamos según los scores
    scores_train = predict_scores_any(sol, x)
    scores_train = np.asarray(scores_train).reshape(-1)    
    In = np.where(scores_train <= 0.001)[0]
    Out = np.where(scores_train > 0.001)[0]
    
    # Graficar
    if ax is None:
        _, ax = plt.subplots()
    
    # Dibujar región de decisión PRIMERO
    contourf = ax.contourf(xx, yy, scores_grid, levels=50, cmap='RdYlBu_r', alpha=0.3)
    
    if colorbar_shown:
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Scores')
    
    # Trazar las fronteras de decisión
    color = 'chocolate'
    scores_xy = scores_grid.reshape(xx.shape)
    contour_line = ax.contour(xx, yy, scores_xy, levels=[0], colors=color, linewidths=2)
    ax.clabel(contour_line, fmt='%.3f', inline=True, fontsize=8)
    
    # Clasificamos puntos In y Out
    ax.scatter(x[In, 0], x[In, 1], s=4, color='b') # Pinto en azul todos los datos usados para entrenar
    ax.scatter(x[Out, 0], x[Out, 1], s=4, color='r') # Marco en rojo los outliers
    
    # Establecer límites explícitos
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'{method} (train) kernel - V.O.= {sol.obj:.4f},\nNu={Nu:.3f}, %anom={anom_frac}, sigma={sol.sigma:.3f}')
    
    plt.tight_layout()
    
    if show_plot_details:
        plt.show()

def DrawBalls(sol, anom_frac, method="svdd", ax=None, show_plot_details=True):
    """
    Visualizes clusters and outliers in a 2D space based on the provided solution object from primal MSVDD model version.
    Parameters:
    -----------
        sol (Solution-like object): An object containing the following attributes:
            - x: A numpy array of shape (n_samples, 2) representing the data points.
            - lbl: A list or array of labels corresponding to the data points (optional).
            - c: A numpy array of shape (p, 2) representing the cluster centers.
            - RR: A list or array of radii for the clusters.
            - p: An integer representing the number of clusters.
            - Nu: A float representing the regularization parameter of the model.
            - obj: A float representing the objective value of the solution.
        anom_frac (float): A float representing the fraction of anomalies in the dataset.
        method (str): Name of the method shown in titles and saved filenames.
        ax (matplotlib.axes.Axes or None): A matplotlib Axes object to draw the plot on. If None, a new Axes object will be created.
        show_plot_details (bool): A boolean indicating whether to display the plot details (default is True).
    Returns:
    --------
        None: This function draws the plot and displays it.
    """
    
    ### Read sol:
    if hasattr(sol, "components"):
        x= sol.data[:sol.ntrain, :]
    else:
        x= sol.x
    Nu = sol.Nu    

    if ax is None:
        _, ax = plt.subplots()
    
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    
    # Clasificamos según los scores
    scores_train = predict_scores_any(sol, x)
    scores_train = np.asarray(scores_train).reshape(-1)
    In= np.where(scores_train <= 0)[0] # Guardo el índice los outliers
    Out= np.where(scores_train > 0)[0] # Guardo el índice los outliers
    
    # Clasificar puntos In y Out
    ax.scatter(x[In, 0], x[In, 1], s=5, color="b", label="Regulares")
    ax.scatter(x[Out, 0], x[Out, 1], s=5, color="r", label="Detectados como outliers")

    draw_primal_spheres(ax, sol)   

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('equal')
    plt.tight_layout()

    plt.title(f'{method} (train) - V.O.= {sol.obj:.4f}, Nu={Nu:.3f}, %anom={anom_frac}')

    if show_plot_details:
        plt.show()

def DrawBalls_test(sol, data_test, y_test, ntest, anom_frac,method="svdd", show_plot_detail=True):
    """
    Visualize MS-SVDD test results in 2D by plotting test samples, marking detected outliers,
    and drawing the model's decision spheres.
    The function:
    - Splits test samples by their ground-truth labels (inliers=+1, outliers=-1).
    - Plots inliers (blue) and outliers (red).
    - Predicts outliers using `predict_scores(sol, data_test.T)` and marks detected outliers (black "x").
    - Draws decision spheres for each component j with R[j] > 0, centered at c[j] with radius sqrt(R[j]).
    - Adds a title summarizing k, ntest, anomaly fraction, and counts of detected vs. real outliers.
    Parameters
    ----------
        sol : object
            Trained SVDD solution with attributes:
            - c: array-like of shape (k, 2), sphere centers in 2D.
            - R: array-like of shape (k,), squared radii (radius^2) per sphere.
        data_test : ndarray of shape (2, n)
            2D test samples, one sample per column.
        y_test : array-like of shape (n,)
            Ground-truth labels for the test set, expected in {+1 (inlier), -1 (outlier)}.
        ntest : int
            Number of test samples (used for display in the figure title).
        anom_frac : float
            Anomaly fraction (used for display in the figure title).
        show_plot_detail : bool, default=True
            If True, display the plot via plt.show(). If False, the figure is created and configured
            but not shown (useful for testing or saving externally).
    Returns
    -------
        None: Creates a Matplotlib figure, adds the plot elements, and optionally displays it.
    Notes
    -----
    - Requires an external function `predict_scores(sol, X)` that returns a 1D array of scores,
        where positive scores indicate outliers. Here `X` is `data_test.T` (shape (n, 2)).
    - Assumes 2D inputs. Accesses only the first two dimensions via indices [0, :] and [1, :].
    - Only spheres with positive R[j] are drawn; radii are computed as sqrt(R[j]).
    - The plot uses labels "Regulares", "Outliers", and "Detectados (SVDD)".
    """
    
    # x = sol.x
    if hasattr(sol, "components"):
        x= sol.data[:sol.ntrain, :]
    else:
        x= sol.x
    
    fig, ax = plt.subplots()
    
    # Separar puntos de test por etiqueta real
    regulares = data_test[:, y_test == 1]
    outliers = data_test[:, y_test == -1]
    
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    
    # Pintar puntos de test
    ax.scatter(regulares[0, :], regulares[1, :], s=20, color='blue', alpha=0.7, linewidths=0.5, label='Regulars (test)')
    ax.scatter(outliers[0, :], outliers[1, :], s=20, color='red', alpha=0.7, linewidths=0.5, label='Outliers (test)')
    
    # Predecir outliers con el modelo
    scores = predict_scores_any(sol, data_test.T)
    scores = np.asarray(scores).reshape(-1)
    y_pred = (scores > 0).flatten()  # True = outlier detectado
    
    # Marcar outliers detectados con cruces negras
    outliers_detectados = data_test[:, y_pred]
    ax.scatter(outliers_detectados[0, :], outliers_detectados[1, :], 
                s=30, marker='x', color='black', linewidths=0.5, 
                label=f'Out detected ({method})', zorder=5)
    
    draw_primal_spheres(ax, sol)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()

    # Título con información del modelo
    n_detected = y_pred.sum()
    n_real = (y_test == -1).sum()

    plt.title(f'{method} (Test), ntest={ntest}, %anom={anom_frac:.2f}\nOut detected: {n_detected}/{len(y_test)} | Out reals: {n_real}/{len(y_test)}')

    if show_plot_detail:
        plt.show()

def DrawCurves_test(sol, data_test, y_test, ntest, anom_frac, sigma, method="svdd", show_plot_detail=True, colorbar_shown=True):
    """
    Visualize the MS-SVDD decision region, boundary, and 2D test samples, highlighting
    true labels and detected outliers.
    This function:
    - Builds a 2D grid over the test data span and uses `predict_scores` to obtain
        anomaly scores on the grid.
    - Draws a filled contour map of the scores and the decision boundary at score = 0.
    - Plots test samples labeled as regular (1) and outlier (-1) in different colors.
    - Marks detected outliers (where score > 0) with black crosses.
    - Overlays the active sphere centers (where corresponding radius R > 0) as cyan stars.
    - Adds a title summarizing model and detection statistics and optionally shows the figure.
    Parameters
    ----------
    sol : object
            Trained MS-SVDD solution with at least:
            - R: iterable of sphere radii (only R[j] > 0 considered active/visible).
            - c: array-like of shape (k, 2) with 2D centers of spheres.
    data_test : array-like of shape (2, n_test)
            2D test samples arranged as rows [x1; x2]. Only the first two features are visualized.
    y_test : array-like of shape (n_test,)
            Ground-truth labels for test samples. Expected values are 1 for regular and -1 for outlier.
    ntest : int
            Number of test samples (displayed in the plot title).
    anom_frac : float
            Fraction of anomalies in the test set (displayed in the plot title).
    sigma : float
            Kernel width (or related hyperparameter) displayed in the plot title; not used in computation here.
    show_plot_detail : bool, default=True
            If True, calls plt.show() to display the plot.
    Notes
    -----
    - This function assumes the availability of a `predict_scores(sol, X)` function in scope,
        which returns anomaly scores for samples X of shape (n_samples, 2). A positive score
        is treated as an outlier.
    - Visualization is limited to two dimensions and uses only the first two components
        of the data.
    - The function modifies the given axes (or newly created ones) and does not return a value.
    Dependencies
    ------------
    - numpy
    - matplotlib
    - A compatible `predict_scores` implementation
    """


    fig, ax = plt.subplots(figsize=(9, 5))


    if hasattr(sol, "components"):
        x= sol.data[:sol.ntrain, :]
    else:
        x= sol.x
            
    # Separar puntos de test por etiqueta real
    regulares = data_test[:, y_test == 1]
    outliers = data_test[:, y_test == -1]

    # Crear rejilla para visualizar la frontera de decisión
    m = 100
    
    x_min, x_max = data_test[0, :].min() - 1, data_test[0, :].max() + 1
    y_min, y_max = data_test[1, :].min() - 1, data_test[1, :].max() + 1 
        
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, m), np.linspace(y_min, y_max, m))
    grid = np.c_[xx.ravel(), yy.ravel()]  # (m², 2)
    
    # Predecir sobre la rejilla
    scores_grid = predict_scores_any(sol, grid)
    scores_grid = scores_grid.reshape(xx.shape)
        
    # Dibujar región de decisión
    contourf = ax.contourf(xx, yy, scores_grid, levels=50, cmap='RdYlBu_r', alpha=0.3)
    if colorbar_shown:
        cbar = fig.colorbar(contourf, ax=ax)
        cbar.set_label('Scores')
    
    # Dibujar frontera de decisión (score = 0)
    contour_line = ax.contour(xx, yy, scores_grid, levels=[0], colors='chocolate', linewidths=2)
    ax.clabel(contour_line, fmt='%.3f', inline=True, fontsize=8)
    
    # Pintar puntos de test
    ax.scatter(regulares[0, :], regulares[1, :], s=20, color='blue', alpha=0.7, 
                linewidths=0.5, label='Regulars (test)')
    ax.scatter(outliers[0, :], outliers[1, :], s=20, color='red', alpha=0.7,
                linewidths=0.5, label='Outliers (test)')
    
    # Predecir outliers con el modelo
    scores_test = predict_scores_any(sol, data_test.T)
    y_pred = scores_test > 0  # True = outlier detectado
        
    # Marcar outliers detectados con cruces negras
    outliers_detectados = data_test[:, y_pred.flatten()]
    ax.scatter(outliers_detectados[0, :], outliers_detectados[1, :], 
                s=30, marker='x', color='black', linewidths=0.5, 
                label=f'Out detected ({method})', zorder=5)

    # Título con información del modelo
    n_detected = y_pred.sum()
    n_real = (y_test == -1).sum()
    
    # Establecer límites explícitos
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')

    ax.set_title(f'{method} (Test) Kernel, σ={sigma:.2f}, ntest={ntest}, %anom={anom_frac:.2f}\nDetectados: {n_detected}/{len(y_test)} | Reales: {n_real}/{len(y_test)}')
    
    ax.legend()
    plt.tight_layout()
    
    if show_plot_detail:
        plt.show()

def DrawSklearnBoundary(sol, data=None, ax=None, m=300, show_scores=True):
    """
    Draws the decision boundary of a fitted sklearn SVM-type model.
    
    Params:
    -------
        sol (SklearnSolution-like object): Object containing the fitted sklearn model and data.
        data (np.ndarray or None): Data used to determine the plotting limits. If None, sol.data is used.
        ax (matplotlib.axes.Axes or None): Axis where the plot is drawn. If None, a new figure is created.
        m (int): Number of grid points per axis used to draw the boundary.
        show_scores (bool): Whether to draw a filled contour map of decision scores.
    
    Returns:
    --------
        matplotlib.axes.Axes: Axis containing the drawn decision boundary.
    """

    if ax is None:
        _, ax = plt.subplots()

    # Usamos los datos completos para definir los límites de dibujo si se pasan.
    if data is not None:
        if data.shape[0] == 2:
            X_plot = data.T
        else:
            X_plot = data
    else:
        X_plot = sol.x

    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, m), np.linspace(y_min, y_max, m))
    grid = np.c_[xx.ravel(), yy.ravel()]
    model_type = getattr(sol, "model_type", None)

    if model_type == "ocsvm":
        scores = -sol.model.decision_function(grid)

    elif model_type == "svm":
        scores = sol.model.decision_function(grid)

    else:
        raise ValueError("DrawSklearnBoundary solo admite sol.model_type='svm' o 'ocsvm'.")

    scores = scores.reshape(xx.shape)

    # Fondo opcional con el valor del score
    if show_scores:
        contourf = ax.contourf(xx, yy, scores, levels=50, cmap='RdYlBu_r', alpha=0.25)
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label("score")

    # Frontera de decisión: score = 0
    contour = ax.contour(xx, yy, scores, levels=[0], colors="chocolate", linewidths=2)

    ax.clabel(contour, fmt="0.0", inline=True, fontsize=8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    title = "SVM" if model_type == "svm" else "One-Class SVM"
    ax.set_title(f"{title} - frontera de decisión")

    return ax

def DrawSklearnBoundary_test(sol, data_test, y_test, ntest, anom_frac, sigma, method="ocsvm", m=300, show_scores=True, show_plot_detail=True):
    """
    Draws the test decision boundary for a fitted sklearn SVM-type model.
    
    Params:
    -------
        sol (SklearnSolution-like object): Object containing the fitted sklearn model and data.
        data_test (np.ndarray): Test data array, either with samples by rows or with shape (2, n_samples).
        y_test (np.ndarray): True test labels, using 1 for regular samples and -1 for anomalies.
        ntest (int): Number of test samples.
        anom_frac (float): Fraction of anomalies used in the experiment.
        sigma (float): Kernel width used in the model, or -1 for a linear model.
        method (str): Name of the sklearn method, usually 'ocsvm' or 'svm'.
        m (int): Number of grid points per axis used to draw the boundary.
        show_scores (bool): Whether to draw a filled contour map of decision scores.
        show_plot_detail (bool): Whether to show the generated figure.
    
    Returns:
    --------
        None: This function draws the test plot and optionally displays it.
    """

    _, ax = plt.subplots(figsize=(9, 5))

    # Convertimos los datos de test a formato (n_samples, 2) si vienen como (2, n)
    if data_test.shape[0] == 2:
        X_test = data_test.T
    else:
        X_test = data_test

    # Límites del dibujo
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, m), np.linspace(y_min, y_max, m))
    grid = np.c_[xx.ravel(), yy.ravel()]

    model_type = getattr(sol, "model_type", None)

    # score <= 0 -> normal
    # score > 0  -> anomalía
    if model_type == "ocsvm":
        scores_grid = -sol.model.decision_function(grid)
    elif model_type == "svm":
        scores_grid = sol.model.decision_function(grid)
    else:
        raise ValueError("DrawSklearnBoundary_test solo admite 'svm' u 'ocsvm'.")

    scores_grid = scores_grid.reshape(xx.shape)

    # Fondo de scores con el mismo estilo que SVDD
    if show_scores:
        absmax = np.max(np.abs(scores_grid))

        contourf = ax.contourf(xx, yy, scores_grid, levels=50, cmap='RdYlBu_r', alpha=0.3)
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label("Scores")

    # Frontera score = 0
    contour_line = ax.contour(xx, yy, scores_grid, levels=[0], colors='chocolate', linewidths=2)
    ax.clabel(contour_line, fmt='%.3f', inline=True, fontsize=8)

    # Separar test en normales y anomalías reales
    regulares = X_test[y_test == 1]
    outliers = X_test[y_test == -1]

    # Dibujar test real
    ax.scatter(regulares[:, 0], regulares[:, 1], s=20, color='blue', alpha=0.7, linewidths=0.5, label='Regulares (test)')
    ax.scatter(outliers[:, 0], outliers[:, 1], s=20, color='red', alpha=0.7, linewidths=0.5, label='Outliers (test)')

    # Predicción sobre test
    if model_type == "ocsvm":
        scores_test = -sol.model.decision_function(X_test)
    else:
        scores_test = sol.model.decision_function(X_test)

    y_pred = scores_test > 0   # True = detectado como anomalía

    # Marcar anomalías
    outliers_detectados = X_test[y_pred]
    ax.scatter(outliers_detectados[:, 0], outliers_detectados[:, 1], s=30, marker='x', color='black', linewidths=0.7, label=f'Out detected ({method})', zorder=5)

    # Título
    n_detected = np.sum(y_pred)
    n_real = np.sum(y_test == -1)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f'{method} (Test) Kernel, σ={sigma:.2f} ntest={ntest}, %anom={anom_frac:.2f}\n Detectados: {n_detected}/{len(y_test)} | Reales: {n_real}/{len(y_test)}')

    ax.legend()
    plt.tight_layout()

    if show_plot_detail:
        plt.show()



def compute_density_weights(X, n_neighbors=10, eps=1e-8, clip=(0.2, 5.0)):
    """
    Computes density-based weights for the training samples.
    Samples in denser regions receive larger weights, while isolated samples receive smaller weights.
    
    Params:
    -------
        X (np.ndarray): Input data with one sample per row.
        n_neighbors (int): Number of nearest neighbors used to estimate local density.
        eps (float): Small constant used to avoid division by zero.
        clip (tuple): Minimum and maximum allowed values for the normalized weights.
    
    Returns:
    --------
        np.ndarray: Array of density weights normalized to have mean equal to one.
    """

    n = X.shape[0]

    if n <= 1:
        return np.ones(n)

    k = min(n_neighbors + 1, n)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    distances = distances[:, 1:] #Quitamos la primera distancia porque es 0

    mean_dist = distances.mean(axis=1)
    density = 1.0 / (mean_dist + eps)
    weights = density / np.mean(density) #Normalizamos
    weights = np.clip(weights, clip[0], clip[1]) # Evita pesos extremos
    weights = weights / np.mean(weights) # Re-normalizamos 

    return weights

def plot_cross_val(fname):
    """
    Plot cross-validated SVDD metrics from a NumPy .npz file and append a summary row to a table file.
    This function:
    - Loads metric arrays from an .npz archive.
    - Draws error-bar curves across the regularization grid (C) for SVDD.
    - Sets labels, legend, grid, and a descriptive title based on model/metric encoded in the filename.
    - Saves the figure as an SVG alongside the input file and displays it.
    - Computes, per method, the maximum mean metric and its corresponding std and regularization value, and appends these to a text table.
    Parameters:
    -----------
        - fname: str or os.PathLike
            Path to the .npz file containing results.
    Returns:
    --------
        None: The function performs plotting and file writing as side effects.
    Expected:
    ---------
    .npz contents
    - 'maucs': 2D float array, shape (len(nus), n_methods)
        Mean metric values per regularization value (rows) and method (columns).
    - 'saucs': 2D float array, same shape as 'maucs'
        Standard deviations corresponding to 'maucs'.
    - 'Nus': 1D float array
        Grid of regularization values Nu.
    - 'outlier_frac': float
        Proportion of anomalies used in the experiments.
    - 'reps': int
        Number of repetitions; used to convert std to standard error via std / sqrt(reps).
    - Optional: presence of a 'Dual' key is used to label the title as "Dual"; otherwise "Prim".
    Raises:
    -------
    - FileNotFoundError: if the input file does not exist.
    - KeyError: if required arrays/keys are missing in the .npz archive.
    - nTest and reps are inferred from underscore-separated numeric tokens in the filename
    - ValueError: if array shapes are inconsistent or the filename format is incompatible
        with the parsing logic.
    Notes:
    ------
    - Assumes 'maucs' and 'saucs' have shape (len(Cs), n_methods).
    - The standard error is computed as saucs / sqrt(reps); ensure reps > 0.
    """
    
    foo = np.load(fname)
    maucs = foo['maucs']
    saucs = foo['saucs']
    Nus = foo['Nus']
    anom_frac = foo['outlier_frac']
    reps = foo['reps']
    
    DorP = 'Dual' if 'Dual' in fname else 'Prim'  # Extract Dual/Prim from filename
    ntest = int(foo['ntest'])
    nrep = int(foo['reps'])
    
    plt.figure()
    
    plt.errorbar(Nus, maucs, saucs/np.sqrt(reps), fmt='-x', color='orange', ecolor='orange', linewidth=2.0, elinewidth=1.0, alpha=0.8)
    plt.xlim((min(Nus)-0.005, max(Nus)+0.005))
    plt.ylim((0, 1.05))
    plt.xticks(Nus, Nus)
    
    plt.grid()

    xlabel_text = r'Regularization parameter $\nu$'
    
    plt.xlabel(xlabel_text, fontsize=10)
    
    plt.ylabel('AUC-ROC', fontsize=10)
    
    # Construct and set the title
    title = f'{DorP} - AUC-ROC (nTest={ntest}, reps={nrep}, %anom={anom_frac})'
    plt.title(title)
    
    plt.savefig(f'{fname[:-4]}.svg', format='svg')
    plt.show()

def gen_new_data(dir_data, ntrain, nval, ntest, anom_frac, reps, data_k=1, random_state=None, data_shape="spherical", is_shifted=False):
    """
    Generates new synthetic data for training, validation, and testing, ensuring a specified fraction of anomalies in the validation set.
    Parameters:
    -----------
        dir_data (str): The directory path where the generated data and plots will be saved.
        ntrain (int): The number of training samples to generate.
        nval (int): The number of validation samples to generate.
        ntest (int): The number of testing samples to generate.
        anom_frac (float): The fraction of anomalies to include in the dataset.
        reps (iterable): An iterable containing the number of repetitions for data generation.
        data_k (int): Number of normal clusters used by the data generator.
        random_state (int or None): Base random seed used to make the generated data reproducible.
        data_shape (str): Shape of the generated normal data, such as 'normal' or 'banana'.
        is_shifted (bool): If true deviates data from origin to point (3,3).
    Output:
    -------
        None: The function saves the generated data and corresponding labels for training, validation, and testing in NumPy format.
    Behavior:
    ------------
    This function performs the following steps:
    1. Generates synthetic data using the `generate_data` function.
    2. Randomly permutes the generated data and labels.
    3. Ensures that the validation set contains at least 10% anomalies.
    4. Plots the training data, highlighting regular and outlier samples.
    5. Saves the generated data and corresponding labels for training, validation, and testing in NumPy format.
    """

    rng = np.random.default_rng(random_state)
    for n in reps:
        train = np.array(range(ntrain), dtype='i')
        val = np.array(range(ntrain, ntrain+nval), dtype='i')
        test = np.array(range(ntrain+nval, ntrain+nval+ntest), dtype='i')
        datapoints = ntrain + nval + ntest    
        if data_shape == "spherical":
            data, y, z = generate_data_k_clusters(datapoints, outlier_frac=anom_frac, dims=2, n_clusters=data_k, random_state=rng)
        elif data_shape == "banana":
            data, y, z = generate_data_banana_clusters(datapoints, outlier_frac=anom_frac, dims=2, n_clusters=data_k, random_state=rng)
        elif data_shape == "dense":
            data, y, z = generate_data_dense_spherical(datapoints, outlier_frac=anom_frac, dims=2, random_state=rng)
        elif data_shape == "elliptical":
            data, y, z = generate_data_elliptical(datapoints, outlier_frac=anom_frac, dims=2, n_clusters=data_k, random_state=rng)

        if is_shifted:
            data_shift=(3.0,3.0)
            data_shift = np.asarray(data_shift, dtype=float).reshape(-1, 1)
            data = data + data_shift

        inds = rng.permutation(range(ntrain+nval+ntest))
        data = data[:, inds]
        y = y[inds]

        # Me aseguro de que haya al menos un 10% de outliers en validación.
        while np.array(y[val] < 0., dtype='i').sum() <= int(np.ceil(0.1 * nval)):
            inds = rng.permutation(range(ntrain+nval+ntest))
            data = data[:, inds]
            y = y[inds]

        # Gráfica de los datos
        out = [i for i in range(data.shape[1]) if y[i] == -1]
        reg = [i for i in range(data.shape[1]) if y[i] == 1]

        _, ax = plt.subplots()
        ax.axis('equal')

        ax.scatter(data[0, reg], data[1, reg], s=2, color='b', alpha=0.5,label='Regulars')
        ax.scatter(data[0, out], data[1, out], s=2, color='r', alpha=0.5,label='Outliers')

        train_reg = [i for i in train if y[i] == 1]
        train_out = [i for i in train if y[i] == -1]

        ax.scatter(data[0, train_reg], data[1, train_reg], marker='o', s=10, color='c',label='Regulars train')
        ax.scatter(data[0, train_out], data[1, train_out], marker='o', s=10, color='m',label='Outliers train')
        ax.legend(loc='lower left')
        plt.savefig(f'{dir_data}/plot_train_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.png')
        plt.show()

        # Guardo los datos para entrenar con nuestro modelo
        np.save(f'{dir_data}/data_train_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy', data[:, train].copy())    # Guardo los datos de train en un archivo para cargarlos en nuestro modelo.
        np.save(f'{dir_data}/clust_train_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy', y[train].copy())    # Guardo la asignacion de cluster de los datos train
        np.save(f'{dir_data}/data_val_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy', data[:, val].copy())    # Guardo los datos de train en un archivo para cargarlos en nuestro modelo.
        np.save(f'{dir_data}/clust_val_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy', y[val].copy())    # Guardo la asignacion de cluster de los datos train
        np.save(f'{dir_data}/data_test_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy', data[:, test].copy())    # Guardo los datos de train en un archivo para cargarlos en nuestro modelo.
        np.save(f'{dir_data}/clust_test_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy', y[test].copy())    # Guardo la asignacion de cluster de los datos train
               

def generate_data(datapoints, outlier_frac=0.1, dims=2):
    """
    Generates synthetic data for a model using two Gaussian distributions and noise around the clusters.
    Parameters:
    -----------
        datapoints (int): The total number of data points to generate.
        outlier_frac (float, optional): The fraction of data points that should be outliers. Default is 0.1.
        dims (int, optional): The number of dimensions for the data points. Default is 2.
    Returns:
    --------
        tuple: A tuple containing:
            - X (numpy.ndarray): A 2D array of shape (dims, datapoints) containing the generated data points.
            - y (numpy.ndarray): A 1D array of shape (datapoints,) containing labels for the data points, where 
                regular points are labeled as 1 and outliers as -1.
    Notes:
    ------
        The function generates a cluster of points from Gaussian distributions centered 
        around (-1, -1). Noise points are generated in a wider range, avoiding the 
        cluster areas to ensure they are considered outliers.
    """
    
    # """Genera datos sintéticos NUESTROS para el modelo con dos distribuciones gaussianas y ruido alrededor de los clusters,
    # sin llegar a las esquinas ul y lr."""
    
    
    X = np.zeros((dims, datapoints))
    y = np.zeros(datapoints)
    
    # Calculate number of points for each distribution
    num_regular = datapoints - int(np.floor(datapoints * outlier_frac))
    num_noise = datapoints - num_regular
    
    # Generate a regular gaussian distribution
    X[:, :num_regular] = 0.5 * np.random.randn(dims, num_regular) + \
                            np.array([0, 0]).reshape((2, 1)).dot(np.ones((1, num_regular)))
    y[:num_regular] = 1
    
    # Generate noise points avoiding the cluster areas but distributed all around
    noise = np.zeros((dims, num_noise))
    for i in range(num_noise):
        while True:
            # Generate candidate noise point in wider range (-3 to 3)
            point = 6.0 * (np.random.rand(dims) - 0.5)  # Range from -3 to 3
            # Check if point is far enough from both clusters
            dist = np.sqrt(np.sum(point**2))
            # Accept point if outside minimum distance from both clusters
            if dist > 1.5:
                noise[:, i] = point
                break
                
    X[:, num_regular:] = noise
    y[num_regular:] = -1
    
    return X, y

def generate_data_elliptical(datapoints, outlier_frac=0.1, dims=2, n_clusters=1, separation=4.0, noise_box=10.0, random_state=None):
    """
    Generates synthetic data with one or more elliptical normal regions and anomalies located outside all of them.

    Parameters:
    -----------
        datapoints (int): Total number of data points to generate.
        outlier_frac (float, optional): Fraction of observations generated as anomalies.
        dims (int, optional): Dimensionality of the generated data. Currently only dims=2 is supported.
        n_clusters (int, optional): Number of elliptical normal clusters to generate.
        separation (float, optional): Distance used to separate the centers of the elliptical clusters.
        noise_box (float, optional): Size of the box used to generate anomalies.
        random_state (int, numpy.random.Generator, or None, optional): Random seed or NumPy random generator used to make the generated data reproducible.
    Returns:
    --------
        X (numpy.ndarray): Array of shape (dims, datapoints) containing the observations.
        y (numpy.ndarray): Array of shape (datapoints,) containing the anomaly labels. Normal observations are labeled as 1 and anomalies as -1.
        z (numpy.ndarray): Array of shape (datapoints,) containing the cluster labels. Normal observations are labeled from 0 to n_clusters-1 and anomalies are labeled as -1.
    """

    if dims != 2:
        raise ValueError("Actualmente esta función está definida para dims=2.")

    if n_clusters < 1:
        raise ValueError("n_clusters debe ser al menos 1.")

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    num_outliers = int(np.floor(datapoints * outlier_frac))
    num_regular = datapoints - num_outliers

    X = np.zeros((dims, datapoints))
    y = np.ones(datapoints)
    z = np.zeros(datapoints, dtype=int)

    sizes = np.full(n_clusters, num_regular // n_clusters)
    sizes[:num_regular % n_clusters] += 1

    covariance = np.array([
        [1.00, 0.00],
        [0.00, 0.10]
    ])

    inv_covariance = np.linalg.inv(covariance)

    if n_clusters == 1:
        centers = np.array([
            [0.0, 0.0]
        ])

    elif n_clusters == 2:
        centers = np.array([
            [-separation / 2, -separation / 2],
            [ separation / 2,  separation / 2]
        ])

    else:
        angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
        centers = separation * np.column_stack((np.cos(angles), np.sin(angles)))

    start = 0

    for j in range(n_clusters):

        n_j = sizes[j]
        end = start + n_j

        regular = rng.multivariate_normal(mean=centers[j], cov=covariance, size=n_j)

        X[:, start:end] = regular.T
        y[start:end] = 1
        z[start:end] = j

        start = end
    if num_outliers > 0:

        outlier_points = np.zeros((dims, num_outliers))

        for i in range(num_outliers):

            while True:

                point = noise_box * (rng.random(dims) - 0.5)
                mahalanobis_squared = []
                for j in range(n_clusters):

                    diff = point - centers[j]

                    distance = (diff.T @ inv_covariance @ diff)

                    mahalanobis_squared.append(distance)

                if np.all(np.array(mahalanobis_squared) > 6.0):
                    outlier_points[:, i] = point
                    break

        X[:, num_regular:] = outlier_points
        y[num_regular:] = -1
        z[num_regular:] = -1

    perm = rng.permutation(datapoints)

    X = X[:, perm]
    y = y[perm]
    z = z[perm]

    return X, y, z
def generate_data_k_clusters(datapoints, outlier_frac=0.1, dims=2, n_clusters=2, cluster_std=0.35, separation=2.5, noise_box=7.0, random_state=None):
    """
    Generates synthetic data with several separated normal clusters and outliers.
    
    Params:
    -------
        datapoints (int): Total number of generated samples.
        outlier_frac (float): Fraction of samples generated as outliers.
        dims (int): Number of dimensions of the generated data.
        n_clusters (int): Number of normal clusters to generate.
        cluster_std (float): Standard deviation of each normal cluster.
        separation (float): Distance used to separate the cluster centers.
        noise_box (float): Size of the box used to generate outliers.
        random_state (int or None): Random seed used to make the generated data reproducible.
    
    Returns:
    --------
        tuple: Tuple (X, y, z), where X contains data, y contains anomaly labels, and z contains cluster labels.
    """

    rng = np.random.default_rng(random_state)

    X = np.zeros((dims, datapoints))
    y = np.zeros(datapoints)
    z = np.zeros(datapoints, dtype=int)

    num_regular = datapoints - int(np.floor(datapoints * outlier_frac))
    num_noise = datapoints - num_regular

    sizes = np.full(n_clusters, num_regular // n_clusters)
    sizes[:num_regular % n_clusters] += 1

    if dims == 2:
        if n_clusters == 1:
            centers = np.array([[0.0, 0.0]])
        elif n_clusters == 2:
            centers = np.array([
                [-separation / 2, -separation / 2],
                [ separation / 2,  separation / 2]
            ])
        else:
            angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
            centers = separation * np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        centers = np.zeros((n_clusters, dims))
        angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
        centers[:, 0] = separation * np.cos(angles)
        centers[:, 1] = separation * np.sin(angles)

    start = 0

    for j in range(n_clusters):
        end = start + sizes[j]

        X[:, start:end] = (
            cluster_std * rng.standard_normal((dims, sizes[j]))
            + centers[j].reshape(dims, 1)
        )

        y[start:end] = 1
        z[start:end] = j

        start = end

    if num_noise > 0:
        noise_points = np.zeros((dims, num_noise))

        min_outlier_dist = 3 * cluster_std

        for i in range(num_noise):
            while True:
                point = noise_box * (rng.random(dims) - 0.5)

                distances_to_centers = np.linalg.norm(centers - point, axis=1)

                if np.min(distances_to_centers) > min_outlier_dist:
                    noise_points[:, i] = point
                    break

        X[:, num_regular:] = noise_points
        y[num_regular:] = -1
        z[num_regular:] = -1

    perm = rng.permutation(datapoints)

    X = X[:, perm]
    y = y[perm]
    z = z[perm]

    return X, y, z
   
def generate_data_banana_clusters(datapoints, outlier_frac=0.1, dims=2, n_clusters=1, banana_length=3, banana_curvature=1.5, noise=0.15, separation=4.0, noise_box=8.0, random_state=None):
    """
    Generates synthetic banana-shaped normal clusters and outliers.
    
    Params:
    -------
        datapoints (int): Total number of generated samples.
        outlier_frac (float): Fraction of samples generated as outliers.
        dims (int): Number of dimensions of the generated data.
        n_clusters (int): Number of banana-shaped normal clusters to generate.
        banana_length (float): Length of each banana-shaped cluster.
        banana_curvature (float): Curvature applied to the banana-shaped clusters.
        noise (float): Gaussian noise added around the banana curves.
        separation (float): Distance used to separate the cluster centers.
        noise_box (float): Size of the box used to generate outliers.
        random_state (int or None): Random seed used to make the generated data reproducible.
    
    Returns:
    --------
        tuple: Tuple (X, y, z), where X contains data, y contains anomaly labels, and z contains cluster labels.
    """

    rng = np.random.default_rng(random_state)
    if dims < 2:
        raise ValueError("Para datos en un conjunto convexo se necesita dims >= 2.")
    num_outliers = int(np.floor(datapoints * outlier_frac))
    num_regular = datapoints - num_outliers

    X = np.zeros((dims, datapoints))
    y = np.ones(datapoints)
    z = np.zeros(datapoints, dtype=int)

    # Reparto de puntos normales entre los conjuntos
    sizes = np.full(n_clusters, num_regular // n_clusters)
    sizes[:num_regular % n_clusters] += 1

    # Centros de las bananas
    if n_clusters == 1:
        centers = np.array([[0.0, 0.0]])
    else:
        angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
        centers = separation * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )

    start = 0

    for j in range(n_clusters):
        n_j = sizes[j]
        end = start + n_j

        # Longitud del conjunto
        t = rng.uniform(-1.0, 1.0, size=n_j)
        # Forma de parábola
        x1 = banana_length * t
        x2 = banana_curvature * (t**2 - 0.5)
        # Ruido
        x1 += noise * rng.standard_normal(n_j)
        x2 += noise * rng.standard_normal(n_j)
        points_2d = np.vstack([x1, x2])
        # Rotación
        theta = 2 * np.pi * j / max(n_clusters, 1)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])

        points_2d = rotation @ points_2d

        # Traslación al centro correspondiente
        points_2d[0, :] += centers[j, 0]
        points_2d[1, :] += centers[j, 1]

        X[0:2, start:end] = points_2d

        # Si dims > 2, añadimos ruido
        if dims > 2:
            X[2:, start:end] = noise * rng.standard_normal((dims - 2, n_j))

        y[start:end] = 1
        z[start:end] = j
        start = end

    # Generar anomalías uniformes
    if num_outliers > 0:
        outlier_points = np.zeros((dims, num_outliers))

        min_outlier_dist = 0.5

        for i in range(num_outliers):
            while True:
                point = noise_box * (rng.random(dims) - 0.5)

                distances_to_normals = np.linalg.norm(
                    X[:, :num_regular].T - point,
                    axis=1
                )

                if np.min(distances_to_normals) > min_outlier_dist:
                    outlier_points[:, i] = point
                    break

        X[:, num_regular:] = outlier_points
        y[num_regular:] = -1
        z[num_regular:] = -1


    # Mezclar normales y anomalías
    perm = rng.permutation(datapoints)

    X = X[:, perm]
    y = y[perm]
    z = z[perm]
    
    return X, y, z

def fit_svdds_by_labels(instance, labels):
    """
    Trains a different SVDD for every cluster given by the parameter 'labels'

    Params:
    -------
        instance (Instance): Instance containing data, parameters, and training weights.
        labels (np.ndarray): Cluster label assigned to each training sample.

    Returns:
    --------
        components (list): List of solutions given by Primal_SVDD(...) or Dual_SVDD(...). One for every valid cluster.
        total_obj (float): Total values of all objective functions.
        total_runtime (float): Total runtime of all the svdd.
    """
    from models import (Primal_SVDD, Dual_SVDD, Cluster_SVDD, Sklearn_SVM)
    
    xtrain = instance.data[:instance.ntrain, :]
    components = []
    total_obj = 0.0
    total_runtime = 0.0

    for j in range(instance.k):
        idx = np.where(labels == j)[0]

        if len(idx) <= 1: #Para evitar clusters con 0 o 1 punto
            continue

        xj = xtrain[idx, :]
        yj = np.ones(len(idx))

        wght = instance.weights[idx]
        sub_instance = Instance(data=xj, ntrain=len(idx), nval=0, ntest=0, y=yj, Nu=instance.Nu, kernel=instance.kernel, sigma=instance.sigma, degree=instance.degree, k=1, max_iter=instance.max_iter, random_state=instance.random_state, weights=wght)
        
        if instance.kernel == "linear":
            solj = Primal_SVDD(sub_instance)
        else:
            solj = Dual_SVDD(sub_instance)

        solj.cluster_id = j #Para almacenar a qué cluster pertenece cada svdd
        solj.original_indices = idx #Para almacenar los índices usados para entrenar dicho svdd

        components.append(solj)

        if solj.obj is not None:
            total_obj += solj.obj
        if solj.runtime is not None:
            total_runtime += solj.runtime

    return components, total_obj, total_runtime

def evaluate(Nus, sigmas, reps, ntrain, ntest, nval, anom_frac, use_kernel, dir_data, data_DorP, output_DorP,method="svdd",data_shape="normal",data_k=1,k=1, max_iter=20, random_state=0, density_weight=False):
    """
    For each repetition, the function loads train/validation/test splits from disk, concatenates them,
    builds an SVDD instance, solves either the Primal-SVDD (linear) or Dual-SVDD (RBF) model, saves the
    serialized solution to disk, and appends a one-line summary to an output text file. If a solution
    pickle already exists, the computation is skipped and the run is logged as "Done".
    Params:
    -------
        Nus (list): Values of the nu parameter to evaluate.
        sigmas (list): Values of the kernel width parameter to evaluate.
        reps (list): Repetition identifiers used in the dataset filenames.
        ntrain (int): Number of training samples.
        ntest (int): Number of test samples.
        nval (int): Number of validation samples.
        anom_frac (float): Fraction of anomalies used in the datasets.
        use_kernel (bool): Whether to solve the dual RBF formulation instead of the primal linear formulation.
        dir_data (str): Directory containing the generated data files.
        data_DorP (str): Directory where fitted solution files are stored.
        output_DorP (str): Directory where text summaries are appended.
        method (str): Model to fit: 'svdd', 'clustersvdd', 'ocsvm', or 'svm'.
        data_shape (str): Shape identifier used in the dataset filenames.
        data_k (int): Number of true normal clusters used when generating the data.
        k (int): Number of spheres or clusters requested by the model.
        max_iter (int): Maximum number of iterations for iterative methods.
        random_state (int): Random seed used by clustering-based methods.
        density_weight (bool): Whether to use density-based weights in SVDD training.
    Side Effects:
    -------------
    - Loads multiple .npy files from disk (dir_data).
    - Appends a line per run to {output_DorP}/salidas_{Prim|Dual}.txt with runtime, objective, nodes,
        radius R, and a status flag: "SO" when `sol.sub_opt` is True, or "Infeasible" if no objective.
    - Saves each computed solution to {data_DorP}/solucion_{...}.pkl.
    - Skips recomputation when the target .pkl already exists, logging "... - Done".
    Assumptions and Data Layout
    ---------------------------
    - data_* arrays have shape (n_features, n_samples) and are concatenable along axis=1.
    - clust_* arrays are 1D and concatenable along the default axis (axis=0).
    - The SVDD Instance is created with samples in rows: data is transposed before passing to Instance.
    - Requires svdd_class.Instance and svdd_models.{Primal_SVDD, Dual_SVDD} to be importable.
    Raises
    ------
        If any expected .npy file is missing.
        If the required SVDD classes cannot be imported.
    Exception
        Propagates any error occurring during model solving or serialization.
    Returns
    -------
        None: The function performs computations and file I/O as side effects.
    """
    from models import (Primal_SVDD, Dual_SVDD, Cluster_SVDD, Sklearn_SVM)

    if use_kernel:
        DorP = 'Dual'
        kernel = 'rbf'
    else:
        DorP = 'Prim'
        kernel = 'linear'
    
    
    with open(f'{output_DorP}/salidas_{DorP}.txt', 'a') as outfile:
        print(f' Train             ntrain  %anom   rep      Nu       sigma       CPU          V.O.       Nodes        Radius', file=outfile)
        
    for n in reps:
        # Cargamos los datos  
        data_train = np.load(f'{dir_data}/data_train_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy')    # Cargo los datos de train en un archivo para cargarlos en nuestro modelo.
        y_train = np.load(f'{dir_data}/clust_train_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy')    # Cargo la asignacion de cluster de los datos train
        data_val = np.load(f'{dir_data}/data_val_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy')    # Cargo los datos de val en un archivo para cargarlos en nuestro modelo.
        y_val = np.load(f'{dir_data}/clust_val_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy')    # Cargo la asignacion de cluster de los datos val
        data_test = np.load(f'{dir_data}/data_test_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy')    # Cargo los datos de test en un archivo para cargarlos en nuestro modelo.
        y_test = np.load(f'{dir_data}/clust_test_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy')    # Cargo la asignacion de cluster de los datos test

        data = np.concatenate((data_train, data_val, data_test), axis=1)
        y = np.concatenate((y_train, y_val, y_test))
            
        for Nu in Nus:
            for sigma in sigmas:
                ###############################################################
                #                       Solve our model                       #
                ###############################################################

                X_all = data.T
                X_train = X_all[:ntrain, :]

                if density_weight:
                    weights = compute_density_weights(X_train)
                else:
                    weights = None

                instance = Instance(
                    data.T,
                    ntrain,
                    nval,
                    ntest,
                    y,
                    Nu,
                    kernel,
                    sigma,
                    k=k,
                    max_iter=max_iter,
                    random_state=random_state,
                    weights=weights
                )

                dw_tag = "_dw" if density_weight else ""
                solution_file = (f'{data_DorP}/solucion_{method}{dw_tag}_{data_shape}_datak{data_k}_k{k}_{ntrain}_{nval}_{ntest}_anom{anom_frac}_rep{n}_nu{Nu}_sigma{sigma}.pkl')                
                print(f'{solution_file}')

                if os.path.exists(solution_file):
                    with open(f'{output_DorP}/salidas_{DorP}.txt', 'a') as outfile:
                        print(f'{solution_file} - Done', file=outfile)
                    continue
                else:  
                    if method == "svdd":
                        if not use_kernel:
                            sol = Primal_SVDD(instance)
                        else:
                            sol = Dual_SVDD(instance)
                    elif method == "clustersvdd":
                        sol = Cluster_SVDD(instance)
                    elif method in ("ocsvm", "svm"):
                        sol = Sklearn_SVM(instance, method=method)
                    else:
                        raise ValueError(f"Método no reconocido: {method}")     

                # Guardar la sol en una archivo .pkl
                with open(solution_file, 'wb') as f:
                    print(sol)
                    pickle.dump(sol, f)
                
                ###############################################################
                #                      Save output data                       #
                ###############################################################
                with open(f'{output_DorP}/salidas_{DorP}.txt', 'a') as outfile:
                    method_name=method.upper()+"-"+DorP
                    if not sol.obj:
                        status = "   INFEASIBLE" if not sol.sub_opt else "         SO"
                        print(f'{method_name:<19s}   {ntrain:4d}    {anom_frac:4.2f}   {n:3d}    {Nu:6.4f}    {sigma:6.3f}  {sol.runtime:9.3f}                              {status}', file=outfile)

                    else:
                        if hasattr(sol, "components"):
                            radii = [round(float(component.R), 4) for component in sol.components]
                            # radii_str = "[" + ", ".join(f"{component.R:.3f}" for component in sol.components) + "]"
                            
                            status = "  "
                            if method == "clustersvdd":
                                print(f'{method_name:<19s} 'f'{ntrain:4d}    {anom_frac:4.2f}   {n:3d}    'f'{Nu:6.4f}    {sigma:6.3f}  'f'{sol.runtime:9.3f}    {sol.obj:11.6f}      'f'k={sol.k}       R={radii}',file=outfile)
                            
                        elif method == "ocsvm" or method =="svm":
                            status = "SO" if sol.sub_opt else "  "
                            print(f'{method_name:<19s} 'f'{ntrain:4d}    {anom_frac:4.2f}   {n:3d}     'f'{Nu:6.4f}    {sigma:6.3f}  'f'{sol.runtime:9.3f}    'f'{"N.A.":>8}    'f'{"N.A.":>8}       'f'{"N.A.":>6}',file=outfile)        
                        else:
                            status = "SO" if sol.sub_opt else "  "
                            print(f'{method_name:<19s} 'f'{ntrain:4d}    {anom_frac:4.2f}   {n:3d}    'f'{Nu:6.4f}    {sigma:6.3f}  'f'{sol.runtime:9.3f}    {sol.obj:11.6f}{int(sol.nodes):8d}   {status}   {sol.R:9.6f}',file=outfile)


def metrics(res_filename, Nus, sigmas, reps, ntrain, ntest, nval, anom_frac, use_kernel, save_plot_detail, show_plot_detail, dir_data, data_DorP, output_DorP, figures_DorP,method="svdd", data_shape="normal",data_k=1,k=1,density_weight=False):
    """
    Calculates AUC-ROC metric for a given model based on training, validation, and test datasets.
    Params:
    -------
        res_filename (str): Path where the metric arrays will be saved as a .npz file.
        Nus (list): Values of the nu parameter evaluated.
        sigmas (list): Values of the kernel width parameter evaluated.
        reps (list): Repetition identifiers used in the dataset filenames.
        ntrain (int): Number of training samples.
        ntest (int): Number of test samples.
        nval (int): Number of validation samples.
        anom_frac (float): Fraction of anomalies used in the datasets.
        use_kernel (bool): Whether the fitted solutions correspond to the dual RBF formulation.
        save_plot_detail (bool): Whether to save detailed plots for each fitted solution.
        show_plot_detail (bool): Whether to display detailed plots for each fitted solution.
        dir_data (str): Directory containing the generated data files.
        data_DorP (str): Directory containing saved solution files.
        output_DorP (str): Directory where metric summaries are written.
        figures_DorP (str): Directory where detailed figures are saved.
        method (str): Model evaluated: 'svdd', 'clustersvdd', 'ocsvm', or 'svm'.
        data_shape (str): Shape identifier used in the dataset filenames.
        data_k (int): Number of true normal clusters used when generating the data.
        k (int): Number of spheres or clusters requested by the model.
        density_weight (bool or str): Whether density weighting was used when fitting the model.
    Returns:
    --------
        None: The function saves the results to the specified file and generates plots if required.
    """
    

    
    if use_kernel:
        DorP = 'Dual'
        cabecera = 'sigma elegido'
    else:
        DorP = 'Prim'
        cabecera = ''
        
    val = np.array(range(ntrain, ntrain+nval), dtype='i')
    test = np.array(range(ntrain+nval, ntrain+nval+ntest), dtype='i')
    
    rocs = []
    recalls = []
    fprs = []
    with open(f'{output_DorP}/salidas_{DorP}.txt', 'a') as outfile:
        print(f' Test              ntrain  %anom   rep      Nu       sigma       CPU          V.O.        Nodes          ROC       Recall       FPR      TP    TN    FP    FN   {cabecera}', file=outfile)
                        
    for n in reps:
        rocs_n = []
        recalls_n = []
        fprs_n = []
        # Load data
        data_train = np.load(f'{dir_data}/data_train_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy')    # Cargo los datos de train en un archivo para cargarlos en nuestro modelo.
        y_train = np.load(f'{dir_data}/clust_train_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy')    # Cargo la asignacion de cluster de los datos train
        data_val = np.load(f'{dir_data}/data_val_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy')    # Cargo los datos de val en un archivo para cargarlos en nuestro modelo.
        y_val = np.load(f'{dir_data}/clust_val_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy')    # Cargo la asignacion de cluster de los datos val
        data_test = np.load(f'{dir_data}/data_test_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy')    # Cargo los datos de test en un archivo para cargarlos en nuestro modelo.
        y_test = np.load(f'{dir_data}/clust_test_{data_shape}_datak{data_k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}.npy')    # Cargo la asignacion de cluster de los datos test



        data = np.concatenate((data_train, data_val, data_test), axis=1)
        y = np.concatenate((y_train, y_val, y_test))
    
        # Relabel outliers as 1 (positive class) and regulars as 0
        y_true = np.where(y == -1, 1, 0)
            
        for Nu in Nus:
            max_roc, max_val_roc, sigma_roc = -1.0, -1.0, -1.0
            recall_at_best_roc = -1.0
            fpr_at_best_roc = -1.0

            TNt, FPt, FNt, TPt = -1, -1, -1, -1
            
            for sigma in sigmas:
                ###############################################################
                #                          Predictions                        #
                ###############################################################
                
                # Check if solution file exists

                dw_tag = "_dw" if density_weight else ""
                solution_file = (f'{data_DorP}/solucion_{method}{dw_tag}_{data_shape}_datak{data_k}_k{k}_{ntrain}_{nval}_{ntest}_anom{anom_frac}_rep{n}_nu{Nu}_sigma{sigma}.pkl')                
                print(f'{solution_file}')
                
                if os.path.exists(solution_file):
                    # Load the results obtained previously and saved in *.pkl files to make the predictions
                    with open(solution_file, 'rb') as f:
                        sol = pickle.load(f)
                else:
                    # Skip to next iteration if the solution file doesn't exist
                    with open(f'{output_DorP}/salidas_{DorP}.txt', 'a') as outfile:
                        print(f'{solution_file} - Not evaluated', file=outfile)
                    continue

                if sol.obj is not None:

                    scores = predict_scores_any(sol, data.T)
                    scores = np.asarray(scores).reshape(-1)

                    y_pred = np.where(scores > 0, 1, 0)


                    # Update ROC scores
                    print(f'Evaluating ROC for n={n}, Nu={Nu}, sigma={sigma}')
                    max_val_roc, max_roc, sigma_roc = calculate_single_metric('roc', scores, y_pred, y_true, val, test, max_val_roc, max_roc, sigma_roc, sigma, use_labels_for_roc_pr=False)

                    # Si calculate_single_metric ha seleccionado este sigma
                    if sigma_roc == sigma:
                        TNt, FPt, FNt, TPt = confusion_matrix(y_true[test], y_pred[test], labels=[0, 1]).ravel()
                        recall_at_best_roc = (TPt / (TPt + FNt) if (TPt + FNt) > 0 else 0.0)
                        fpr_at_best_roc = (FPt / (FPt + TNt) if (FPt + TNt) > 0 else 0.0)
                    
                ###############################################################
                #                      Save output data                       #
                ###############################################################

                with open(f'{output_DorP}/salidas_{DorP}.txt', 'a') as outfile:
                    sigmas_choosen = f"{sigma_roc:.2f}   " if use_kernel else ""
                    method_name=method.upper()+"-"+DorP
                    if hasattr(sol, "components"):
                        if sol.obj is None:
                            status = "   Infactible"
                            print(f'{method_name:<19s} 'f'{ntrain:4d}    {anom_frac:4.2f}   {n:3d}    'f'{Nu:6.4f}    {sigma:6.3f}  'f'{sol.runtime:9.3f}                              {status}',file=outfile)

                        else:
                            sub_opt = any(getattr(component, "sub_opt", False) for component in sol.components)
                            nodes = sum(getattr(component, "nodes", 0) for component in sol.components)


                            status = "SO" if sub_opt else "  "

                            print(f'{method_name:<19s} 'f'{ntrain:4d}    {anom_frac:4.2f}   {n:3d}    'f'{Nu:6.4f}    {sigma:6.3f}  'f'{sol.runtime:9.3f}    {sol.obj:9.6f}   'f'{int(nodes):8d}   {status}     {max_roc:5.6f}   {recall_at_best_roc:5.6f}   {fpr_at_best_roc:5.6f}   {TPt:3d}   {TNt:3d}   {FPt:3d}   {FNt:3d}     'f'{sigmas_choosen}',file=outfile)
                    else:
                        if sol.obj is None:
                            status = "   Infactible" if not sol.sub_opt else "         SO"
                            print(f'{method_name:<19s} 'f'{ntrain:4d}    {anom_frac:4.2f}   {n:3d}    'f'{Nu:6.4f}    {sigma:6.3f}  'f'{sol.runtime:9.3f}                              {status}',file=outfile)
                        else:
                            status = "SO" if sol.sub_opt else "  "
                            print(f'{method_name:<19s} 'f'{ntrain:4d}    {anom_frac:4.2f}   {n:3d}    'f'{Nu:6.4f}    {sigma:6.3f}  'f'{sol.runtime:9.3f}     {sol.obj:9.6f}  'f'{int(sol.nodes):8d}   {status}    {max_roc:5.6f}    {recall_at_best_roc:5.6f}   {fpr_at_best_roc:5.6f}   {TPt:3d}   {TNt:3d}   {FPt:3d}   {FNt:3d}     'f'{sigmas_choosen}',file=outfile) 
                ###############################################################
                #                       Draw the results                      #
                ###############################################################
                # Draw the datapoints by colors according to whether they are regular or outliers     
                if show_plot_detail or save_plot_detail: # If there exists a solution, draw our results
                    fig, ax = plt.subplots(figsize=(9, 5))
                    if sol.obj is not None:
                        reg = [i for i in range(data.shape[1]) if y[i] == 1] # Renamed to avoid conflict with outer loop 'r'
                        out = [i for i in range(data.shape[1]) if y[i] == -1] # Renamed to avoid conflict

                        ax.scatter(data[0, reg], data[1, reg], s=2, color='c', alpha=0.5)
                        ax.scatter(data[0, out], data[1, out], s=2, color='orange', alpha=0.5)
                    
                        if getattr(sol, "model_type", None) in ("svm", "ocsvm"):
                            DrawSklearnBoundary(sol, data=data, ax=ax, m=300, show_scores=True)
                            DrawSklearnBoundary_test(sol, data_test, y_test, ntest, anom_frac, sigma, method=method, m=300, show_scores=True, show_plot_detail=show_plot_detail)

                        elif not use_kernel:
                            DrawBalls(sol, anom_frac, method, ax, show_plot_detail)
                            DrawBalls_test(sol, data_test, y_test, ntest, anom_frac, method, show_plot_detail=show_plot_detail)

                        else:
                            DrawCurves(sol, anom_frac, method, ax, show_plot_detail, colorbar_shown=True)
                            DrawCurves_test(sol, data_test, y_test, ntest, anom_frac, sigma, method, show_plot_detail=show_plot_detail,colorbar_shown=True)
                    
                    set_global_plot_limits(ax, data, padding=0.05)

                    if show_plot_detail:
                        plt.show()
                        
                    if save_plot_detail:
                        fig.savefig(f'{figures_DorP}/plot_{method}{dw_tag}_{data_shape}_datak{data_k}_k{k}_ntr{ntrain}_nval{nval}_nte{ntest}_anom{anom_frac}_rep{n}_Nu{Nu}_sigma{sigma}.png')
                    plt.close(fig)
                            
            # rocs[reps.index(n), Nus.index(Nu)] = max_roc

            rocs_n.append(max_roc)
            recalls_n.append(recall_at_best_roc)
            fprs_n.append(fpr_at_best_roc)
        rocs.append(rocs_n)
        recalls.append(recalls_n)
        fprs.append(fprs_n)
    mrocs = np.mean(rocs, axis=0)
    srocs = np.std(rocs, axis=0)

    mrecalls = np.mean(recalls, axis=0)
    srecalls = np.std(recalls, axis=0)

    mfprs = np.mean(fprs, axis=0)
    sfprs = np.std(fprs, axis=0)
    
    # save results
    np.savez(res_filename, maucs=mrocs, saucs=srocs, mrecalls=mrecalls, srecalls=srecalls, mfprs=mfprs, sfprs=sfprs, Nus=Nus, ntrain=ntrain, ntest=ntest, nval=nval, outlier_frac=anom_frac, reps=len(reps), sigmas=sigmas)
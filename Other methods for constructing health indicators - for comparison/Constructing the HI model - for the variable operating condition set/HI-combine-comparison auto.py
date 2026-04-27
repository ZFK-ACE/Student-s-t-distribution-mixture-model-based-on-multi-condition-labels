import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import special, stats
from scipy.signal import medfilt
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import os
import glob

warnings.filterwarnings('ignore')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ======================================================================================
# PART 1: Classes from "Construction of multi-condition health indicators..."
# ======================================================================================

class KalmanFilter1D:
    """
    1D Kalman Filter (from Construction... file)
    """

    def __init__(self, initial_value=0.0, process_variance=1e-4, measurement_variance=1.0):
        self.x = initial_value
        self.P = 1.0
        self.Q = process_variance
        self.R = measurement_variance
        self.F = 1.0
        self.H = 1.0

    def update(self, measurement):
        # Predict
        self.x = self.F * self.x
        self.P = self.F * self.P * self.F + self.Q
        # Update
        y = measurement - self.H * self.x
        S = self.H * self.P * self.H + self.R
        K = self.P * self.H / S
        self.x = self.x + K * y
        self.P = (1 - K * self.H) * self.P
        return self.x

    def filter_sequence(self, measurements):
        filtered = np.zeros_like(measurements)
        for i, z in enumerate(measurements):
            filtered[i] = self.update(z)
        return filtered


class TDistributionHealthIndicator:
    """
    Health Indicator Builder based on t-distribution mixture model (from Construction... file)
    """

    def __init__(self, n_components=3, covariance_type='full', nu=5.0,
                 random_state=42, use_pca=True, n_components_pca=None, variance_threshold=0.95,
                 kalman_process_variance=1e-4, kalman_measurement_variance=0.1, scale_factor=25.0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.nu = nu
        self.random_state = random_state
        self.use_pca = use_pca
        self.n_components_pca = n_components_pca
        self.variance_threshold = variance_threshold
        self.kalman_process_variance = kalman_process_variance
        self.kalman_measurement_variance = kalman_measurement_variance

        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.healthy_log_likelihood_mean = None
        self.healthy_log_likelihood_std = None
        self.healthy_log_likelihood_threshold = None
        self.kalman_filter = None
        self.scale_factor = scale_factor  # 新增属性：接收外部传入的 scale_factor

    def fit(self, X_healthy):
        # Standardize
        if isinstance(X_healthy, pd.DataFrame):
            X_scaled = self.scaler.fit_transform(X_healthy.values)
        else:
            X_scaled = self.scaler.fit_transform(X_healthy)

        # PCA
        if self.use_pca:
            X_scaled = self._apply_pca(X_scaled)

        # Create Model
        self._create_model(X_scaled.shape[1])
        self.model.fit(X_scaled)

        # Calculate Healthy Stats
        log_likelihood_healthy = self._calculate_t_log_likelihood(X_scaled)
        self.healthy_log_likelihood_mean = np.mean(log_likelihood_healthy)
        self.healthy_log_likelihood_std = np.std(log_likelihood_healthy)
        self.healthy_log_likelihood_threshold = np.percentile(log_likelihood_healthy, 0.5)

        return self

    def _create_model(self, n_features):
        degrees_of_freedom_prior = max(n_features + 1, 30.0)
        self.model = BayesianGaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.1,
            mean_precision_prior=0.1,
            degrees_of_freedom_prior=degrees_of_freedom_prior,
            random_state=self.random_state,
            max_iter=200,
            n_init=3
        )

    def _apply_pca(self, X_scaled):
        if self.n_components_pca is None:
            self.pca = PCA(n_components=self.variance_threshold, random_state=self.random_state)
        else:
            n_components = min(self.n_components_pca, X_scaled.shape[1])
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
        return self.pca.fit_transform(X_scaled)

    def _calculate_t_log_likelihood(self, X):
        n_samples, n_features = X.shape
        log_likelihood = np.zeros(n_samples)
        weights = self.model.weights_
        means = self.model.means_
        covariances = self.model.covariances_

        for k in range(len(weights)):
            if weights[k] < 1e-6: continue
            X_centered = X - means[k]

            if self.covariance_type == 'full':
                cov = covariances[k]
                reg = 1e-6 * np.eye(cov.shape[0])
                try:
                    inv_cov = np.linalg.pinv(cov + reg)
                    _, logdet = np.linalg.slogdet(cov + reg)
                    mahalanobis = np.sum(X_centered @ inv_cov * X_centered, axis=1)
                    log_det_sigma = logdet
                except:
                    diag_cov = np.diag(cov)
                    mahalanobis = np.sum(X_centered ** 2 / (diag_cov + 1e-6), axis=1)
                    log_det_sigma = np.sum(np.log(diag_cov + 1e-6))
            elif self.covariance_type == 'diag':
                mahalanobis = np.sum(X_centered ** 2 / (covariances[k] + 1e-6), axis=1)
                log_det_sigma = np.sum(np.log(covariances[k] + 1e-6))
            else:  # spherical
                mahalanobis = np.sum(X_centered ** 2, axis=1) / (covariances[k] + 1e-6)
                log_det_sigma = n_features * np.log(covariances[k] + 1e-6)

            d = n_features
            nu = self.nu
            log_const = (special.gammaln((nu + d) / 2) - special.gammaln(nu / 2)
                         - (d / 2) * np.log(nu * np.pi))
            log_prob_k = (log_const - 0.5 * log_det_sigma
                          - ((nu + d) / 2) * np.log(1 + mahalanobis / nu))

            if k == 0:
                log_likelihood = np.log(weights[k] + 1e-300) + log_prob_k
            else:
                max_log = np.maximum(log_likelihood, np.log(weights[k] + 1e-300) + log_prob_k)
                log_likelihood = max_log + np.log(
                    np.exp(log_likelihood - max_log) +
                    np.exp(np.log(weights[k] + 1e-300) + log_prob_k - max_log)
                )
        return log_likelihood

    def calculate_single_sample_hi(self, x_new):
        """
        Modified method to calculate raw HI for a single sample without internal smoothing
        """
        # Reshape for single sample
        x_new = x_new.reshape(1, -1)

        # Transform
        x_scaled = self.scaler.transform(x_new)
        if self.use_pca and self.pca is not None:
            x_scaled = self.pca.transform(x_scaled)

        # Log Likelihood
        log_likelihood = self._calculate_t_log_likelihood(x_scaled)[0]

        # Calculate HI
        log_likelihood_diff = self.healthy_log_likelihood_mean - log_likelihood
        # 使用动态传入的 scale_factor 替换固定的数值
        scaled_diff = log_likelihood_diff / (self.scale_factor * self.healthy_log_likelihood_std)
        health_indicator = np.exp(-scaled_diff)
        health_indicator = np.clip(health_indicator, 0, 1)

        return health_indicator, log_likelihood


# ======================================================================================
# PART 2: Clustering Method and CLEANING UTILS
# ======================================================================================

def bic_based_clustering(data, n_components_range=range(1, 12), covariance_type='tied'):
    """
    BIC-based clustering (Low sensitivity mode)
    """
    best_model = None
    best_bic = np.inf
    best_n = 0
    results = {}

    print("\n正在使用BIC准则选择最优簇数量 (低敏感度模式)...")

    bic_values = []

    for n in n_components_range:
        model = GaussianMixture(
            n_components=n,
            covariance_type=covariance_type,
            random_state=42,
            max_iter=300,
            n_init=10,
            reg_covar=0.3,
            tol=1e-3
        )
        model.fit(data)
        labels = model.predict(data)
        bic = model.bic(data)
        bic_values.append(bic)

        results[n] = {
            'model': model,
            'labels': labels,
            'bic': bic,
            'converged': model.converged_,
            'n_iter': model.n_iter_
        }
        print(f"簇数={n:2d}, BIC={bic:.2f}")

        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_n = n

    if best_model is None and len(results) > 0:
        best_n = min(results.keys())
        best_model = results[best_n]['model']

    return best_model, best_n, results, bic_values


def filter_short_segments(labels, min_segment_length=50):
    """
    【噪声簇过滤机制】
    """
    n = len(labels)
    if n == 0: return labels

    cleaned_labels = labels.copy()

    # 1. 识别所有连续片段: (label, start_index, end_index)
    segments = []
    if n > 0:
        current_label = labels[0]
        start = 0
        for i in range(1, n):
            if labels[i] != current_label:
                segments.append({'label': current_label, 'start': start, 'end': i})
                current_label = labels[i]
                start = i
        segments.append({'label': current_label, 'start': start, 'end': n})

    # 2. 过滤逻辑
    last_stable_label = segments[0]['label']
    changes_count = 0

    for i, seg in enumerate(segments):
        length = seg['end'] - seg['start']
        if length < min_segment_length and i > 0:
            cleaned_labels[seg['start']: seg['end']] = last_stable_label
            changes_count += 1
        else:
            last_stable_label = seg['label']

    if changes_count > 0:
        print(f"[Noise Filter] 已合并 {changes_count} 个噪点簇 (长度 < {min_segment_length}) 到相邻稳定簇中。")
    else:
        print("[Noise Filter] 未发现需要过滤的噪点簇。")

    return cleaned_labels


# ======================================================================================
# PART 3: Data Loading and Processing
# ======================================================================================

def load_combined_data(file_path):
    """
    Load data looking for specific Current and Force columns.
    Adaptive to missing channels.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Path {file_path} not found. Searching current directory.")
        file_path = "."

    found_files = []
    if os.path.isdir(file_path):
        for ext in ['*.csv', '*.xlsx', '*.xls', '*.txt']:
            found_files.extend(glob.glob(os.path.join(file_path, ext)))
        if not found_files:
            print("Error: No data files found.")
            return None, None
        target_file = sorted(found_files, key=os.path.getsize, reverse=True)[0]
    else:
        target_file = file_path

    print(f"Loading file: {target_file}")
    try:
        if target_file.lower().endswith('.csv'):
            for enc in ['utf-8', 'gbk', 'gb2312']:
                try:
                    df = pd.read_csv(target_file, encoding=enc)
                    break
                except:
                    continue
        elif target_file.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(target_file)
        else:
            df = pd.read_csv(target_file, delimiter='\t')
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None

    # Column Mapping
    col_map = {}
    for axis in ['u', 'v', 'w']:
        candidates = [c for c in df.columns if
                      f'cur_{axis}' in c.lower() or f'current_{axis}' in c.lower() or f'i_{axis}' in c.lower()]
        if candidates: col_map[f'Cur_{axis}'] = candidates[0]

    for axis in ['x', 'y', 'z']:
        candidates = [c for c in df.columns if f'f_{axis}' in c.lower() or f'force_{axis}' in c.lower()]
        if candidates: col_map[f'F_{axis}'] = candidates[0]

    found_cluster_cols = [k for k in col_map.keys() if 'Cur' in k]
    found_hi_cols = [k for k in col_map.keys() if 'F_' in k]

    if not found_cluster_cols:
        print("Error: No Current columns (Cur_u, v, w) found. Cannot perform clustering.")
        return None, None
    if not found_hi_cols:
        print("Error: No Force columns (F_x, y, z) found. Cannot calculate HI.")
        return None, None

    df_renamed = df.rename(columns=col_map)
    final_cols = found_cluster_cols + found_hi_cols
    print(f"Data Loaded. Found Channels -> Clustering: {found_cluster_cols}, HI: {found_hi_cols}")

    return df_renamed[final_cols], target_file


def split_and_featurize(data, chunk_size=20000):
    """
    Split data into chunks and calculate features.
    """
    n_samples = len(data) // chunk_size
    print(f"Splitting data into {n_samples} samples (Chunk size: {chunk_size})")

    cur_cols = [c for c in data.columns if c.startswith('Cur')]
    force_cols = [c for c in data.columns if c.startswith('F_')]

    cluster_feats = []
    hi_feats = []

    for i in range(n_samples):
        chunk = data.iloc[i * chunk_size: (i + 1) * chunk_size]
        # Clustering: RMS
        cur_rms = np.sqrt(np.mean(chunk[cur_cols].values ** 2, axis=0))
        cluster_feats.append(cur_rms)
        # HI: Mean
        f_mean = np.mean(chunk[force_cols].values, axis=0)
        hi_feats.append(f_mean)

    return pd.DataFrame(cluster_feats, columns=cur_cols), \
        pd.DataFrame(hi_feats, columns=force_cols)


# ======================================================================================
# PART 4: Visualization Functions
# ======================================================================================

def visualize_clustering(original_rms_data, labels, pca_obj, pca_data):
    """
    Visualizations for clustering
    """
    available_channels = original_rms_data.columns.tolist()
    n_features = len(available_channels)

    fig = plt.figure(figsize=(20, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # 1. RMS Time Series
    for i, channel in enumerate(available_channels):
        if i >= 3: break
        ax = fig.add_subplot(3, 3, i + 1)
        ax.plot(original_rms_data[channel], color='gray', alpha=0.3)
        scatter = ax.scatter(range(len(original_rms_data)), original_rms_data[channel],
                             c=labels, cmap='tab20', s=15)
        ax.set_title(f'{channel} RMS Time Series (Filtered)')
        if i == 0: plt.colorbar(scatter, ax=ax, label='Cluster')

    # 2. Feature Space
    if n_features >= 3:
        ax = fig.add_subplot(3, 3, 4, projection='3d')
        ax.scatter(original_rms_data.iloc[:, 0], original_rms_data.iloc[:, 1], original_rms_data.iloc[:, 2],
                   c=labels, cmap='tab20', s=20)
        ax.set_title('3D Clustering Feature Space')
    elif n_features == 2:
        ax = fig.add_subplot(3, 3, 4)
        ax.scatter(original_rms_data.iloc[:, 0], original_rms_data.iloc[:, 1],
                   c=labels, cmap='tab20', s=20)
        ax.set_title('2D Clustering Feature Space')
    else:
        ax = fig.add_subplot(3, 3, 4)
        ax.scatter(range(len(original_rms_data)), original_rms_data.iloc[:, 0],
                   c=labels, cmap='tab20', s=20)
        ax.set_title('1D Feature Space')

    # 3. PCA
    ax = fig.add_subplot(3, 3, 5)
    if pca_data.shape[1] >= 2:
        ax.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='tab20', s=20)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    else:
        ax.scatter(range(len(pca_data)), pca_data[:, 0], c=labels, cmap='tab20', s=20)
    ax.set_title('PCA Projection')

    # 4. Distribution
    ax = fig.add_subplot(3, 3, 6)
    counts = pd.Series(labels).value_counts().sort_index()
    ax.bar(counts.index, counts.values)
    ax.set_title('Cluster Distribution (After Filtering)')

    plt.tight_layout()
    plt.show()


def visualize_hi(hi_values, log_likelihoods, labels):
    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(hi_values, 'b-', label='Health Indicator')
    ax1.axhline(1.0, color='g', linestyle='--')
    ax1.set_title('Health Indicator over Time')
    ax1.set_ylim(0, 1.1)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    scatter = ax2.scatter(range(len(log_likelihoods)), log_likelihoods, c=labels, cmap='tab20', s=10)
    ax2.set_title('Log Likelihood')
    plt.colorbar(scatter, ax=ax2)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(hi_values, bins=50, color='blue', alpha=0.7)
    ax3.set_title('HI Distribution')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(labels, 'k-', alpha=0.5, drawstyle='steps-post')
    ax4.set_title('Cluster Labels over Time (Filtered)')

    plt.tight_layout()
    plt.show()


def visualize_original_hi_separate(hi_data):
    plt.figure(figsize=(15, 6))
    plt.plot(hi_data, color='#1f77b4', linewidth=1.5, label='Original HI')
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.6)
    plt.axhline(y=0.3, color='red', linestyle=':', alpha=0.6)
    plt.title('Original Health Indicator')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_hi_with_cluster_boundaries(hi_data, labels):
    transitions = np.where(labels[:-1] != labels[1:])[0] + 1
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(111)
    ax.plot(hi_data, color='#1f77b4', linewidth=1.5, label='Health Indicator')
    if len(transitions) > 0:
        ax.axvline(x=transitions[0], color='orange', linestyle='--', alpha=0.8, label='Cluster Boundary')
        for t in transitions[1:]:
            ax.axvline(x=t, color='orange', linestyle='--', alpha=0.8)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.6)
    ax.axhline(y=0.3, color='red', linestyle=':', alpha=0.6)
    ax.set_title('Health Indicator with Cluster Boundaries')
    ax.legend(loc='best')
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.show()


def visualize_fusion_weights(weight_kalman, weight_cumulative):
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    ax.plot(weight_kalman, label='Weight: Kalman', color='blue', alpha=0.8)
    ax.plot(weight_cumulative, label='Weight: Cumulative', color='green', linestyle='--', alpha=0.8)
    ax.set_title('Fusion Weights Variation')
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.show()


def visualize_fused_hi_median_filtered(hi_fused, hi_fused_filtered):
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(111)
    ax.plot(hi_fused, 'r-', alpha=0.3, label='Original Fused HI')
    ax.plot(hi_fused_filtered, 'b-', linewidth=2, label='Median Filtered Fused HI')
    ax.axhline(0.3, color='k', linestyle=':')
    ax.set_title('Fused Health Indicator with Median Filtering')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.show()


def visualize_cumulative_hi_only(health_indicator_cumulative, anomaly_flags):
    fig = plt.figure(figsize=(15, 8))
    cumulative_anomalies = np.cumsum(anomaly_flags)
    ax1 = fig.add_subplot(111)
    ax1.plot(health_indicator_cumulative, 'r-', linewidth=2, label='Cumulative HI')
    ax1.set_ylabel('Health Indicator (0-1)', color='r')
    ax1.set_ylim(0, 1.1)
    ax2 = ax1.twinx()
    ax2.plot(cumulative_anomalies, 'g--', alpha=0.6, label='Cumulative Count')
    ax2.set_ylabel('Count', color='g')
    plt.title('Cumulative Anomaly Health Indicator')
    plt.tight_layout()
    plt.show()


def visualize_final_fusion(hi_kalman, hi_cumulative, hi_fused, fusion_confidences):
    fig = plt.figure(figsize=(18, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(hi_kalman, 'b-', alpha=0.4, linewidth=1, label='Real-time HI')
    ax1.plot(hi_cumulative, 'g--', alpha=0.6, linewidth=1, label='Cumulative HI')
    ax1.plot(hi_fused, 'r-', linewidth=2, label='Fused HI')
    ax1.axhline(0.3, color='k', linestyle=':')
    ax1.set_title('Multi-Indicator Fusion Comparison')
    ax1.legend()
    ax1.set_ylim(0, 1.1)

    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.plot(fusion_confidences, color='purple', alpha=0.7)
    ax2.fill_between(range(len(fusion_confidences)), 0, fusion_confidences, color='purple', alpha=0.1)
    ax2.set_title('Fusion Confidence')
    ax2.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.show()


# ======================================================================================
# PART 5: Fusion Logic
# ======================================================================================

def build_cumulative_anomaly_health_indicator(anomaly_flags, health_indicator_kalman=None):
    n_samples = len(anomaly_flags)
    cumulative_anomalies = np.cumsum(anomaly_flags)
    max_cumulative = np.max(cumulative_anomalies)
    if max_cumulative > 0:
        health_indicator_cumulative = 1 - (cumulative_anomalies / max_cumulative)
    else:
        health_indicator_cumulative = np.ones(n_samples)
    return health_indicator_cumulative


def decision_level_fusion_optimized(hi_kalman, hi_cumulative,
                                    threshold_low=0.3,
                                    threshold_high=0.7,
                                    consistency_threshold=0.2):
    diff = abs(hi_kalman - hi_cumulative)

    def sigmoid(x, center=0.0, slope=3.0):
        try:
            val = 1.0 / (1.0 + np.exp(-slope * (x - center)))
        except OverflowError:
            val = 0.0 if slope * (x - center) > 0 else 1.0
        return val

    if hi_kalman < threshold_low and hi_cumulative < threshold_low:
        min_val = min(hi_kalman, hi_cumulative)
        min_bias = sigmoid(1.0 - hi_kalman, center=0.5, slope=3.0) * sigmoid(1.0 - hi_cumulative, center=0.5, slope=3.0)
        hi_fused = min_bias * min_val + (1 - min_bias) * (0.5 * hi_kalman + 0.5 * hi_cumulative)
        if hi_kalman < hi_cumulative:
            weight_kalman = 0.5 + 0.3 * min_bias
            weight_cumulative = 0.5 - 0.3 * min_bias
        else:
            weight_kalman = 0.5 - 0.3 * min_bias
            weight_cumulative = 0.5 + 0.3 * min_bias
        fusion_type = "Crisis-Min"
        confidence = 0.8 + 0.1 * min_bias

    elif hi_kalman > threshold_high and hi_cumulative > threshold_high:
        max_val = max(hi_kalman, hi_cumulative)
        max_bias = sigmoid(hi_kalman, center=0.5, slope=3.0) * sigmoid(hi_cumulative, center=0.5, slope=3.0)
        hi_fused = max_bias * max_val + (1 - max_bias) * (0.5 * hi_kalman + 0.5 * hi_cumulative)
        if hi_kalman > hi_cumulative:
            weight_kalman = 0.5 + 0.3 * max_bias
            weight_cumulative = 0.5 - 0.3 * max_bias
        else:
            weight_kalman = 0.5 - 0.3 * max_bias
            weight_cumulative = 0.5 + 0.3 * max_bias
        fusion_type = "Healthy-Max"
        confidence = 0.8 + 0.1 * max_bias

    elif diff < consistency_threshold:
        consistency_factor = 1.0 - (diff / consistency_threshold)
        base_weight = 0.5
        adjustment = 0.1 * consistency_factor
        weight_kalman = base_weight + adjustment
        weight_cumulative = base_weight - adjustment
        total_weight = weight_kalman + weight_cumulative
        weight_kalman /= total_weight
        weight_cumulative /= total_weight
        hi_fused = weight_kalman * hi_kalman + weight_cumulative * hi_cumulative
        fusion_type = "Consistent-Avg"
        confidence = 0.7 + 0.2 * consistency_factor

    else:
        cumulative_low_factor = sigmoid(threshold_low - hi_cumulative, center=0.0, slope=2.5)
        kalman_low_factor = sigmoid(threshold_low - hi_kalman, center=0.0, slope=2.5)
        base_weight_kalman = 0.6
        base_weight_cumulative = 0.4
        if hi_cumulative < threshold_low:
            if hi_kalman > threshold_high:
                adjustment = 0.25 * (1.0 - cumulative_low_factor)
                weight_kalman = base_weight_kalman + adjustment
                weight_cumulative = base_weight_cumulative - adjustment
            else:
                adjustment = 0.15 * cumulative_low_factor
                weight_kalman = 0.55 + adjustment
                weight_cumulative = 0.45 - adjustment
        elif hi_kalman < threshold_low:
            if hi_cumulative > threshold_high:
                adjustment = 0.25 * (1.0 - kalman_low_factor)
                weight_kalman = base_weight_kalman - adjustment
                weight_cumulative = base_weight_cumulative + adjustment
            else:
                adjustment = 0.15 * kalman_low_factor
                weight_kalman = 0.55 - adjustment
                weight_cumulative = 0.45 + adjustment
        else:
            diff_factor = sigmoid(diff, center=0.5, slope=3.0)
            adjustment = 0.15 * diff_factor
            weight_kalman = 0.6 + adjustment
            weight_cumulative = 0.4 - adjustment

        weight_kalman = max(0.15, min(0.85, weight_kalman))
        weight_cumulative = 1.0 - weight_kalman
        hi_fused = weight_kalman * hi_kalman + weight_cumulative * hi_cumulative
        confidence = 0.6 - 0.3 * min(diff, 0.5)
        fusion_type = "Conflict-Adjusted"

    hi_fused = max(0.0, min(1.0, hi_fused))
    weight_sum = weight_kalman + weight_cumulative
    weight_kalman /= weight_sum
    weight_cumulative /= weight_sum

    return hi_fused, fusion_type, confidence, weight_kalman, weight_cumulative


# ======================================================================================
# PART 6: Main Logic
# ======================================================================================

def main():
    print("=" * 60)
    print("Integrated HI Construction System (With Noise Filtering)")
    print("=" * 60)

    # 1. Read Data
    data_path = r"E:\铣刀数据\2023_7_铣刀\训练数据"
    df, source_file_path = load_combined_data(data_path)

    # ==========================================================
    # 【新增：保存路径处理】
    # ==========================================================
    base_save_path = r"E:\铣刀数据\2023_7_铣刀\comparison of models\构建的HI\MY—HI"
    if source_file_path:
        file_name = os.path.basename(source_file_path).split('.')[0]
        final_output_dir = os.path.join(base_save_path, file_name)
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir)
            print(f"Created directory: {final_output_dir}")
    else:
        final_output_dir = base_save_path
    # ==========================================================

    if df is None:
        print("Creating Mock Data for Demonstration...")
        np.random.seed(42)
        n_points = 20000 * 500
        t = np.linspace(0, 100, n_points)
        regime = np.zeros(n_points)
        regime[n_points // 3: 2 * n_points // 3] = 1
        regime[2 * n_points // 3:] = 2
        data = {
            'Cur_u': np.sin(t) + np.random.normal(0, 0.1, n_points) + regime,
            'Cur_w': np.sin(t * 2) + np.random.normal(0, 0.1, n_points) + regime * 0.5,
            'F_x': np.random.normal(10, 1, n_points) + regime * 2,
            'F_z': np.random.normal(10, 1, n_points) + regime * 4
        }
        df = pd.DataFrame(data)

    # --- 新增：在特征提取前，从原始数据前 2,000,0 个样本中计算全局 scale_factor ---
    force_cols_raw = [c for c in ['F_x', 'F_y', 'F_z'] if c in df.columns]
    global_scale_factor = 25.0  # 默认值
    if force_cols_raw:
        f_max_abs = np.max(np.abs(df[force_cols_raw].iloc[:20000].values))
        print(f"f_max_abs = {f_max_abs}")
        if f_max_abs > 0:
            global_scale_factor = 300 / f_max_abs
    # -------------------------------------------------------------------------

    # 2. Split and Featurize
    df_cluster_feats, df_hi_feats = split_and_featurize(df, chunk_size=20000)

    if len(df_cluster_feats) < 10:
        print("Not enough data samples.")
        return

    # 3. Clustering
    scaler_cluster = StandardScaler()
    cluster_data_scaled = scaler_cluster.fit_transform(df_cluster_feats)

    model_cluster, best_n, results, bic_vals = bic_based_clustering(cluster_data_scaled)
    labels_raw = model_cluster.predict(cluster_data_scaled)

    # ==============================================================================
    # PART 3.5: Filter Noise Clusters
    # ==============================================================================
    print("\nFiltering small cluster segments (Noise Removal)...")
    noise_threshold = 50
    labels = filter_short_segments(labels_raw, min_segment_length=noise_threshold)
    # ==============================================================================

    # PCA for visualization
    n_cluster_feats = cluster_data_scaled.shape[1]
    pca_cluster = PCA(n_components=min(2, n_cluster_feats))
    pca_data = pca_cluster.fit_transform(cluster_data_scaled)

    print("Generating Clustering Visualizations...")
    visualize_clustering(df_cluster_feats, labels, pca_cluster, pca_data)

    # 4. HI Construction Loop (Using CLEANED labels)
    print("\nConstructing Health Indicators...")

    hi_models = {}
    cluster_history = {l: [] for l in np.unique(labels)}
    cluster_indices = {l: [] for l in np.unique(labels)}
    cluster_offsets = {}

    global_kf = KalmanFilter1D(initial_value=1.0, process_variance=1e-4, measurement_variance=0.1)
    n_construct_samples = 100

    n_total_samples = len(df_hi_feats)
    raw_hi_array = np.full(n_total_samples, np.nan)
    final_hi_stitched_array = np.full(n_total_samples, np.nan)
    log_likelihood_array = np.full(n_total_samples, np.nan)
    thresholds_array = np.full(n_total_samples, -np.inf)
    current_offset = 0.0

    for i in range(n_total_samples):
        current_label = labels[i]
        current_force = df_hi_feats.iloc[i].values

        cluster_history[current_label].append(current_force)
        cluster_indices[current_label].append(i)
        current_count = len(cluster_history[current_label])

        if current_count < n_construct_samples:
            continue
        elif current_count == n_construct_samples:
            baseline_data = np.array(cluster_history[current_label])
            # 修改处：实例化时将全局计算的 scale_factor 传入
            model = TDistributionHealthIndicator(n_components=min(3, len(baseline_data)), covariance_type='diag', scale_factor=global_scale_factor)
            try:
                model.fit(baseline_data)
                hi_models[current_label] = model
                stored_indices = cluster_indices[current_label]
                first_idx = stored_indices[0]
                prev_idx = first_idx - 1

                batch_raw_his = []
                batch_log_likelihoods = []
                for val in baseline_data:
                    rh, ll = model.calculate_single_sample_hi(val)
                    batch_raw_his.append(rh)
                    batch_log_likelihoods.append(ll)

                avg_raw_start = np.mean(batch_raw_his)
                if prev_idx < 0:
                    prev_final_hi = 1.0
                else:
                    prev_val = final_hi_stitched_array[prev_idx]
                    prev_final_hi = prev_val if not np.isnan(prev_val) else 1.0

                current_offset = avg_raw_start - prev_final_hi
                cluster_offsets[current_label] = current_offset

                for k in range(n_construct_samples):
                    hist_idx = stored_indices[k]
                    rh = batch_raw_his[k]
                    ll = batch_log_likelihoods[k]
                    raw_hi_array[hist_idx] = rh
                    log_likelihood_array[hist_idx] = ll
                    thresholds_array[hist_idx] = model.healthy_log_likelihood_threshold
                    stitched_val = rh - current_offset
                    final_hi_stitched_array[hist_idx] = np.clip(stitched_val, 0.0, 1.0)
            except Exception as e:
                print(f"Error fitting model: {e}")
        else:
            if current_label in hi_models:
                if i > 0 and labels[i] != labels[i - 1]:
                    if current_label in cluster_offsets:
                        current_offset = cluster_offsets[current_label]
                rh, ll = hi_models[current_label].calculate_single_sample_hi(current_force)
                raw_hi_array[i] = rh
                log_likelihood_array[i] = ll
                thresholds_array[i] = hi_models[current_label].healthy_log_likelihood_threshold
                stitched_val = rh - current_offset
                final_hi_stitched_array[i] = np.clip(stitched_val, 0.0, 1.0)
            else:
                raw_hi_array[i] = 1.0
                final_hi_stitched_array[i] = 1.0

    final_hi_stitched_array[np.isnan(final_hi_stitched_array)] = 1.0
    final_hi_filtered = global_kf.filter_sequence(final_hi_stitched_array)

    anomaly_flags = []
    for i in range(n_total_samples):
        thresh = thresholds_array[i] if thresholds_array[i] != -np.inf else -1e9
        is_anomaly = (log_likelihood_array[i] < thresh) or (final_hi_filtered[i] < 0.3)
        anomaly_flags.append(is_anomaly)
    anomaly_flags = np.array(anomaly_flags)

    visualize_hi(final_hi_filtered, log_likelihood_array, labels)

    # 5. Fusion
    hi_cumulative_arr = build_cumulative_anomaly_health_indicator(anomaly_flags, final_hi_filtered)

    hi_fused_list, conf_list, w_k_list, w_c_list = [], [], [], []
    for k in range(len(final_hi_filtered)):
        hi_f, _, conf, w_k, w_c = decision_level_fusion_optimized(final_hi_filtered[k], hi_cumulative_arr[k])
        hi_fused_list.append(hi_f)
        conf_list.append(conf)
        w_k_list.append(w_k)
        w_c_list.append(w_c)

    hi_fused_arr = np.array(hi_fused_list)
    hi_fused_filtered = medfilt(hi_fused_arr, kernel_size=211)

    # ==========================================================
    # 【核心新增：保存 Origin 数据】
    # ==========================================================
    print(f"\nSaving data for Origin to: {final_output_dir}")

    # 1. 保存聚类与特征数据 (RMS, PCA, Labels)
    origin_cluster_df = df_cluster_feats.copy()
    origin_cluster_df['Labels_Filtered'] = labels
    origin_cluster_df['PCA_1'] = pca_data[:, 0]
    origin_cluster_df['PCA_2'] = pca_data[:, 1]
    origin_cluster_df.to_csv(os.path.join(final_output_dir, "Origin_Clustering_Data.csv"), index=False)

    # 2. 保存 HI 与 融合数据
    origin_hi_df = pd.DataFrame({
        'Sample_Index': range(n_total_samples),
        'Raw_HI': raw_hi_array,
        'Stitched_HI': final_hi_stitched_array,
        'Kalman_Filtered_HI': final_hi_filtered,
        'Log_Likelihood': log_likelihood_array,
        'Threshold': thresholds_array,
        'Cumulative_HI': hi_cumulative_arr,
        'Anomaly_Flag': anomaly_flags.astype(int),
        'Fused_HI_Raw': hi_fused_arr,
        'Fused_HI_Median_Filtered': hi_fused_filtered,
        'Fusion_Confidence': conf_list,
        'Weight_Kalman': w_k_list,
        'Weight_Cumulative': w_c_list,
        'Cluster_Label': labels
    })
    origin_hi_df.to_csv(os.path.join(final_output_dir, "Origin_HI_Fusion_Results.csv"), index=False)

    print("Origin CSV files saved successfully.")
    # ==========================================================

    # 保持原有的所有可视化输出
    visualize_original_hi_separate(final_hi_filtered)
    visualize_hi_with_cluster_boundaries(final_hi_filtered, labels)
    visualize_cumulative_hi_only(hi_cumulative_arr, anomaly_flags)
    visualize_final_fusion(final_hi_filtered, hi_cumulative_arr, hi_fused_arr, np.array(conf_list))
    visualize_fusion_weights(np.array(w_k_list), np.array(w_c_list))
    visualize_fused_hi_median_filtered(hi_fused_arr, hi_fused_filtered)

    print("Processing Complete.")


if __name__ == "__main__":
    main()
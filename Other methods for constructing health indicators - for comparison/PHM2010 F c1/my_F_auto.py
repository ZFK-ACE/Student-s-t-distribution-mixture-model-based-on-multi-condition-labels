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

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ======================================================================================
# PART 1: 健康指标构建类 (保持不变)
# ======================================================================================

class KalmanFilter1D:
    def __init__(self, initial_value=0.0, process_variance=1e-4, measurement_variance=1.0):
        self.x = initial_value
        self.P = 1.0
        self.Q = process_variance
        self.R = measurement_variance
        self.F = 1.0
        self.H = 1.0

    def update(self, measurement):
        self.x = self.F * self.x
        self.P = self.F * self.P * self.F + self.Q
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
    def __init__(self, n_components=3, covariance_type='full', nu=5.0,
                 random_state=42, use_pca=True, n_components_pca=None, variance_threshold=0.95,
                 kalman_process_variance=1e-4, kalman_measurement_variance=0.1, scale_factor=25):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.nu = nu
        self.random_state = random_state
        self.use_pca = use_pca
        self.n_components_pca = n_components_pca
        self.variance_threshold = variance_threshold
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.healthy_log_likelihood_mean = None
        self.healthy_log_likelihood_std = None
        self.healthy_log_likelihood_threshold = None
        self.scale_factor = scale_factor  # 新增属性：接收外部传入的 scale_factor

    def fit(self, X_healthy):
        if isinstance(X_healthy, pd.DataFrame):
            X_scaled = self.scaler.fit_transform(X_healthy.values)
        else:
            X_scaled = self.scaler.fit_transform(X_healthy)

        if self.use_pca:
            X_scaled = self._apply_pca(X_scaled)

        self._create_model(X_scaled.shape[1])
        self.model.fit(X_scaled)

        log_likelihood_healthy = self._calculate_t_log_likelihood(X_scaled)
        self.healthy_log_likelihood_mean = np.mean(log_likelihood_healthy)
        self.healthy_log_likelihood_std = np.std(log_likelihood_healthy)
        self.healthy_log_likelihood_threshold = np.percentile(log_likelihood_healthy, 10)
        return self

    def _create_model(self, n_features):
        df_prior = max(n_features + 1, 30.0)
        self.model = BayesianGaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.1,
            mean_precision_prior=0.1,
            degrees_of_freedom_prior=df_prior,
            random_state=self.random_state,
            max_iter=200, n_init=3
        )

    def _apply_pca(self, X_scaled):
        if self.n_components_pca is None:
            self.pca = PCA(n_components=self.variance_threshold, random_state=self.random_state)
        else:
            n_comp = min(self.n_components_pca, X_scaled.shape[1])
            self.pca = PCA(n_components=n_comp, random_state=self.random_state)
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
                cov = covariances[k] + 1e-6 * np.eye(covariances[k].shape[0])
                inv_cov = np.linalg.pinv(cov)
                _, logdet = np.linalg.slogdet(cov)
                mahalanobis = np.sum(X_centered @ inv_cov * X_centered, axis=1)
                log_det_sigma = logdet
            elif self.covariance_type == 'diag':
                mahalanobis = np.sum(X_centered ** 2 / (covariances[k] + 1e-6), axis=1)
                log_det_sigma = np.sum(np.log(covariances[k] + 1e-6))
            else:
                mahalanobis = np.sum(X_centered ** 2, axis=1) / (covariances[k] + 1e-6)
                log_det_sigma = n_features * np.log(covariances[k] + 1e-6)

            nu = self.nu
            log_const = (special.gammaln((nu + n_features) / 2) - special.gammaln(nu / 2)
                         - (n_features / 2) * np.log(nu * np.pi))
            log_prob_k = (log_const - 0.5 * log_det_sigma
                          - ((nu + n_features) / 2) * np.log(1 + mahalanobis / nu))

            if k == 0:
                log_likelihood = np.log(weights[k] + 1e-300) + log_prob_k
            else:
                max_log = np.maximum(log_likelihood, np.log(weights[k] + 1e-300) + log_prob_k)
                log_likelihood = max_log + np.log(np.exp(log_likelihood - max_log) +
                                                  np.exp(np.log(weights[k] + 1e-300) + log_prob_k - max_log))
        return log_likelihood

    def calculate_single_sample_hi(self, x_new):
        x_new = x_new.reshape(1, -1)
        x_scaled = self.scaler.transform(x_new)
        if self.use_pca and self.pca is not None:
            x_scaled = self.pca.transform(x_scaled)
        log_likelihood = self._calculate_t_log_likelihood(x_scaled)[0]
        log_likelihood_diff = self.healthy_log_likelihood_mean - log_likelihood
        # 使用动态传入的 scale_factor 替换固定的 25
        scaled_diff = log_likelihood_diff / (self.scale_factor * self.healthy_log_likelihood_std)
        health_indicator = np.exp(-scaled_diff)
        return np.clip(health_indicator, 0, 1), log_likelihood


# ======================================================================================
# PART 2: 聚类与过滤工具 (保持不变)
# ======================================================================================

def bic_based_clustering(data, n_components_range=range(1, 12), covariance_type='tied'):
    best_model, best_bic, best_n, results = None, np.inf, 0, {}
    print("\n正在使用BIC准则选择最优簇数量...")
    bic_values = []
    for n in n_components_range:
        model = GaussianMixture(n_components=n, covariance_type=covariance_type, random_state=42,
                                max_iter=300, n_init=10, reg_covar=3, tol=1e-3)
        model.fit(data)
        bic = model.bic(data)
        bic_values.append(bic)
        results[n] = {'model': model, 'labels': model.predict(data), 'bic': bic}
        print(f"簇数={n:2d}, BIC={bic:.2f}")
        if bic < best_bic:
            best_bic, best_model, best_n = bic, model, n
    return best_model, best_n, results, bic_values


def filter_short_segments(labels, min_segment_length=50):
    n = len(labels)
    if n == 0: return labels
    cleaned_labels = labels.copy()
    segments = []
    current_label, start = labels[0], 0
    for i in range(1, n):
        if labels[i] != current_label:
            segments.append({'label': current_label, 'start': start, 'end': i})
            current_label, start = labels[i], i
    segments.append({'label': current_label, 'start': start, 'end': n})
    last_stable_label, changes_count = segments[0]['label'], 0
    for i, seg in enumerate(segments):
        if (seg['end'] - seg['start']) < min_segment_length and i > 0:
            cleaned_labels[seg['start']: seg['end']] = last_stable_label
            changes_count += 1
        else:
            last_stable_label = seg['label']
    print(f"[Noise Filter] 合并了 {changes_count} 个噪点簇。" if changes_count > 0 else "[Noise Filter] 未发现噪点簇。")
    return cleaned_labels


# ======================================================================================
# PART 3: 数据加载与处理 (保持不变)
# ======================================================================================

def load_combined_data(file_path):
    if not os.path.exists(file_path):
        file_path = "."
    found_files = []
    if os.path.isdir(file_path):
        for ext in ['*.csv', '*.xlsx', '*.xls', '*.txt']:
            found_files.extend(glob.glob(os.path.join(file_path, ext)))
        if not found_files: return None
        target_file = sorted(found_files, key=os.path.getsize, reverse=True)[0]
    else:
        target_file = file_path

    print(f"Loading file: {target_file}")
    try:
        if target_file.lower().endswith('.csv'):
            df = pd.read_csv(target_file, encoding='gbk')
        elif target_file.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(target_file)
        else:
            df = pd.read_csv(target_file, delimiter='\t')
    except:
        return None

    col_map = {}
    ae_cols_in_df = [c for c in df.columns if 'ae' in c.lower()]

    if not ae_cols_in_df:
        print("错误: 未在文档中找到 AE 列。")
        return None

    base_ae_col = ae_cols_in_df[0]
    df['AE_1'] = df[base_ae_col]
    df['AE_2'] = df[base_ae_col]
    df['AE_3'] = df[base_ae_col]
    cluster_cols = ['AE_1', 'AE_2', 'AE_3']

    for axis in ['x', 'y', 'z']:
        candidates = [c for c in df.columns if f'f_{axis}' in c.lower() or f'force_{axis}' in c.lower()]
        if candidates: col_map[f'F_{axis}'] = candidates[0]

    hi_cols = [k for k in col_map.keys() if 'F_' in k]
    if not hi_cols:
        print("错误: 未找到 Force 列。")
        return None

    df_renamed = df.rename(columns=col_map)
    print(f"数据加载成功。聚类通道: {cluster_cols}, HI通道: {hi_cols}")
    return df_renamed[cluster_cols + hi_cols]


def split_and_featurize(data, chunk_size=20000):
    n_samples = len(data) // chunk_size
    ae_cols = [c for c in data.columns if c.startswith('AE')]
    force_cols = [c for c in data.columns if c.startswith('F_')]
    cluster_feats, hi_feats = [], []
    for i in range(n_samples):
        chunk = data.iloc[i * chunk_size: (i + 1) * chunk_size]
        cluster_feats.append(np.sqrt(np.mean(chunk[ae_cols].values ** 2, axis=0)))
        hi_feats.append(np.mean(chunk[force_cols].values, axis=0))
    return pd.DataFrame(cluster_feats, columns=ae_cols), pd.DataFrame(hi_feats, columns=force_cols)


# ======================================================================================
# PART 4: 可视化与融合逻辑 (保持不变)
# ======================================================================================

def visualize_clustering(original_rms_data, labels, pca_obj, pca_data):
    fig = plt.figure(figsize=(20, 15))
    available_channels = original_rms_data.columns.tolist()
    for i, channel in enumerate(available_channels[:3]):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.plot(original_rms_data[channel], color='gray', alpha=0.3)
        ax.scatter(range(len(original_rms_data)), original_rms_data[channel], c=labels, cmap='tab20', s=15)
        ax.set_title(f'{channel} RMS Time Series')
    ax4 = fig.add_subplot(3, 3, 4, projection='3d')
    ax4.scatter(original_rms_data.iloc[:, 0], original_rms_data.iloc[:, 1], original_rms_data.iloc[:, 2], c=labels,
                cmap='tab20', s=20)
    ax4.set_title('3D Clustering Space')
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='tab20', s=20)
    ax5.set_title('PCA Projection')
    plt.tight_layout()
    plt.show()


def visualize_hi(hi_values, log_likelihoods, labels):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes[0, 0].plot(hi_values, 'b-');
    axes[0, 0].set_title('Health Indicator')
    axes[0, 1].scatter(range(len(log_likelihoods)), log_likelihoods, c=labels, cmap='tab20', s=10);
    axes[0, 1].set_title('Log Likelihood')
    axes[1, 0].hist(hi_values, bins=50);
    axes[1, 0].set_title('HI Distribution')
    axes[1, 1].plot(labels, 'k-', alpha=0.5);
    axes[1, 1].set_title('Labels')
    plt.tight_layout();
    plt.show()


def visualize_original_hi_separate(hi_data):
    plt.figure(figsize=(15, 6))
    plt.plot(hi_data);
    plt.axhline(1.0, color='g', ls='--');
    plt.axhline(0.3, color='r', ls=':')
    plt.title('Final Stitched HI');
    plt.ylim(0, 1.1);
    plt.show()


def visualize_hi_with_cluster_boundaries(hi_data, labels):
    transitions = np.where(labels[:-1] != labels[1:])[0] + 1
    plt.figure(figsize=(18, 6))
    plt.plot(hi_data)
    for t in transitions: plt.axvline(x=t, color='orange', alpha=0.5)
    plt.title('HI with Boundaries');
    plt.ylim(0, 1.1);
    plt.show()


def visualize_final_fusion(hi_kalman, hi_cumulative, hi_fused, conf):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
    ax1.plot(hi_kalman, 'b', alpha=0.3, label='Real-time')
    ax1.plot(hi_cumulative, 'g', alpha=0.3, label='Cumulative')
    ax1.plot(hi_fused, 'r', lw=2, label='Fused')
    ax1.legend();
    ax2.plot(conf, color='purple');
    ax2.set_title('Confidence')
    plt.tight_layout();
    plt.show()


# --- 新增的可视化函数 ---
def visualize_final_hi_standalone(hi_fused):
    """
    单独展示融合后的最终健康指标
    """
    plt.figure(figsize=(15, 6))
    plt.plot(hi_fused, color='crimson', linewidth=2.5, label='Fused Health Indicator')
    plt.fill_between(range(len(hi_fused)), hi_fused, color='crimson', alpha=0.1)
    plt.axhline(1.0, color='green', linestyle='--', alpha=0.6, label='Perfect Health')
    plt.axhline(0.3, color='red', linestyle=':', alpha=0.8, label='Failure Threshold')
    plt.title('Final Integrated Health Indicator (Decision Fusion Result)', fontsize=14)
    plt.xlabel('Samples (Chunks)', fontsize=12)
    plt.ylabel('Health Score', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')
    plt.ylim(-0.05, 1.05)
    plt.show()


def build_cumulative_anomaly_health_indicator(anomaly_flags):
    cum_anomalies = np.cumsum(anomaly_flags)
    max_cum = np.max(cum_anomalies)
    return 1 - (cum_anomalies / max_cum) if max_cum > 0 else np.ones(len(anomaly_flags))


def decision_level_fusion_optimized(hi_k, hi_c):
    diff = abs(hi_k - hi_c)
    hi_f = 0.6 * hi_k + 0.4 * hi_c
    conf = 1 - diff
    return hi_f, conf


# ======================================================================================
# PART 6: 主程序
# ======================================================================================

def main():
    data_path = r"E:\铣刀数据\2023_7_铣刀\训练数据"
    df = load_combined_data(data_path)
    if df is None: return

    # --- 新增：在特征提取前，从原始数据前 2,000,000 个样本中计算全局 scale_factor ---
    force_cols_raw = [c for c in ['F_x', 'F_y', 'F_z'] if c in df.columns]
    global_scale_factor = 25  # 默认值
    if force_cols_raw:
        f_max_abs = np.max(np.abs(df[force_cols_raw].iloc[:100000].values))
        print(f"f_max_abs = {f_max_abs}")
        if f_max_abs > 0:
            global_scale_factor = 300 / f_max_abs
    # -------------------------------------------------------------------------

    # --- 为了获取 source_file_path 用于保存文件夹命名 (不修改 load 函数返回值) ---
    if os.path.isdir(data_path):
        found_files = []
        for ext in ['*.csv', '*.xlsx', '*.xls', '*.txt']:
            found_files.extend(glob.glob(os.path.join(data_path, ext)))
        if found_files:
            source_file_path = sorted(found_files, key=os.path.getsize, reverse=True)[0]
        else:
            source_file_path = None
    else:
        source_file_path = data_path
    # -------------------------------------------------------------------------

    # ==========================================================
    # 【新增：保存路径处理 (仿照参考代码)】
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

    df_cluster_feats, df_hi_feats = split_and_featurize(df)

    # 聚类
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_cluster_feats)
    model, n, _, _ = bic_based_clustering(scaled_data)
    labels = filter_short_segments(model.predict(scaled_data))

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data) # 提取变量以便保存
    visualize_clustering(df_cluster_feats, labels, pca, pca_data)

    # HI 构建
    n_total = len(df_hi_feats)
    hi_array, ll_array, thresholds = np.ones(n_total), np.zeros(n_total), np.full(n_total, -1e9)
    hi_models = {}

    for i in range(n_total):
        label = labels[i]
        feat = df_hi_feats.iloc[i].values
        if label not in hi_models:
            # 修改处：实例化时将全局计算的 scale_factor 传入
            m = TDistributionHealthIndicator(scale_factor=global_scale_factor).fit(df_hi_feats[labels == label].iloc[:100])
            hi_models[label] = m
        hi, ll = hi_models[label].calculate_single_sample_hi(feat)
        hi_array[i], ll_array[i] = hi, ll
        thresholds[i] = hi_models[label].healthy_log_likelihood_threshold # 更新阈值以便保存

    kf = KalmanFilter1D(initial_value=1.0)
    hi_filtered = kf.filter_sequence(hi_array)

    visualize_hi(hi_filtered, ll_array, labels)
    visualize_original_hi_separate(hi_filtered)
    visualize_hi_with_cluster_boundaries(hi_filtered, labels)

    # 融合
    anomalies = hi_filtered < 0.6
    hi_cum = build_cumulative_anomaly_health_indicator(anomalies)
    fused_results = [decision_level_fusion_optimized(hi_filtered[k], hi_cum[k]) for k in range(n_total)]
    hi_fused = np.array([r[0] for r in fused_results]) # 转为numpy数组
    conf = [r[1] for r in fused_results]

    # 为了匹配保存格式，计算中值滤波后的HI
    hi_fused_filtered = medfilt(hi_fused, kernel_size=211)

    # ==========================================================
    # 【核心新增：保存 Origin 数据 (仿照参考代码)】
    # ==========================================================
    print(f"\nSaving data for Origin to: {final_output_dir}")

    # 1. 保存聚类与特征数据 (RMS, PCA, Labels)
    origin_cluster_df = df_cluster_feats.copy()
    origin_cluster_df['Labels_Filtered'] = labels
    origin_cluster_df['PCA_1'] = pca_data[:, 0]
    origin_cluster_df['PCA_2'] = pca_data[:, 1]
    origin_cluster_df.to_csv(os.path.join(final_output_dir, "Origin_Clustering_Data.csv"), index=False)

    # 2. 保存 HI 与 融合数据
    # 注意：本代码融合权重固定为 0.6/0.4，为保持格式一致，手动填充这两列
    origin_hi_df = pd.DataFrame({
        'Sample_Index': range(n_total),
        'Raw_HI': hi_array,
        'Stitched_HI': hi_array, # 本代码无拼接逻辑，用 Raw_HI 替代
        'Kalman_Filtered_HI': hi_filtered,
        'Log_Likelihood': ll_array,
        'Threshold': thresholds,
        'Cumulative_HI': hi_cum,
        'Anomaly_Flag': anomalies.astype(int),
        'Fused_HI_Raw': hi_fused,
        'Fused_HI_Median_Filtered': hi_fused_filtered,
        'Fusion_Confidence': conf,
        'Weight_Kalman': [0.6] * n_total,
        'Weight_Cumulative': [0.4] * n_total,
        'Cluster_Label': labels
    })
    origin_hi_df.to_csv(os.path.join(final_output_dir, "Origin_HI_Fusion_Results.csv"), index=False)

    print("Origin CSV files saved successfully.")
    # ==========================================================

    visualize_final_fusion(hi_filtered, hi_cum, hi_fused, conf)

    # 新增：单独展示融合后的 HI
    visualize_final_hi_standalone(hi_fused)

    print("处理完成。")


if __name__ == "__main__":
    main()
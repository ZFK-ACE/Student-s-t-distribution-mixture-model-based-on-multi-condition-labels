import os
import glob
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import butter, filtfilt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HealthIndicatorBuilder:
    def __init__(self, window_size=50):
        # 恢复 window_size 用于均值滤波的窗口大小
        self.window_size = int(window_size)

    def build_hi_from_residuals(self, residuals, method='exponential'):
        """
        将残差映射为归一化健康指标，并进行均值滤波处理。
        返回范围 [0.0, 1.0] 的平滑后数组。
        """
        residuals = np.asarray(residuals, dtype=float)
        if residuals.size == 0:
            return np.array([])

        # 1. 原始映射逻辑
        if method == 'exponential':
            denom = np.percentile(residuals, 90)
            if denom <= 0:
                denom = np.max(residuals) + 1e-10
            hi_raw = np.exp(-residuals / denom)
        elif method == 'sigmoid':
            rmin, rmax = np.min(residuals), np.max(residuals)
            denom = (rmax - rmin) if (rmax - rmin) > 0 else 1.0
            residual_norm = (residuals - rmin) / denom
            hi_raw = 1 - 1 / (1 + np.exp(-10 * (residual_norm - 0.5)))
        elif method == 'linear':
            rmin, rmax = np.min(residuals), np.max(residuals)
            denom = (rmax - rmin) if (rmax - rmin) > 0 else 1.0
            hi_raw = 1 - (residuals - rmin) / denom
        else:
            raise ValueError(f"未知的方法: {method}")

        # --- 新增：均值滤波处理 (Moving Average Filter) ---
        if self.window_size > 1:
            # 使用 numpy.convolve 实现均值滤波，'same' 保证输出长度不变
            window = np.ones(self.window_size) / self.window_size
            hi_smoothed = np.convolve(hi_raw, window, mode='same')

            # 处理卷积造成的边缘效应：用原始值的首尾进行填充，防止边缘大幅掉落
            pad_size = self.window_size // 2
            hi_smoothed[:pad_size] = hi_raw[:pad_size]
            hi_smoothed[-pad_size:] = hi_raw[-pad_size:]
            hi_final = hi_smoothed
        else:
            hi_final = hi_raw

        # 限幅处理
        hi_final = np.clip(hi_final, 0.0, 1.0)
        return hi_final

    def detect_anomalies(self, hi, threshold=0.3):
        hi = np.asarray(hi, dtype=float)
        if hi.size == 0:
            return np.array([], dtype=bool)
        return hi < float(threshold)

    def calculate_hi_trend(self, hi):
        hi = np.asarray(hi, dtype=float)
        if hi.size == 0:
            return 0.0, "无数据"
        x = np.arange(len(hi))
        slope, intercept = np.polyfit(x, hi, 1)
        trend = slope * len(hi)
        if trend < -0.2:
            level = "快速下降"
        elif trend < -0.1:
            level = "缓慢下降"
        elif trend < 0.1:
            level = "稳定"
        elif trend < 0.2:
            level = "缓慢上升"
        else:
            level = "快速上升"
        return float(trend), level


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_filename = "dataset"  # 新增：用于存储文件名
        self.data = None
        self.features = None
        self.samples = None
        self.sample_features = None
        self.sample_channels = None
        self.num_samples = None
        self.points_per_sample = None

    def load_data(self):
        try:
            if os.path.isfile(self.file_path):
                if self.file_path.lower().endswith('.csv'):
                    # 获取文件名
                    self.data_filename = os.path.basename(self.file_path)

                    df = pd.read_csv(self.file_path)
                    self.data = df.reset_index(drop=True)
                    print(f"加载文件: `{self.file_path}`，形状: {self.data.shape}")
                    return True
                else:
                    return False
            elif os.path.isdir(self.file_path):
                csv_files = glob.glob(os.path.join(self.file_path, "*.csv"))
                if not csv_files: return False

                # 排序并使用第一个CSV文件名作为标识
                csv_files.sort()
                self.data_filename = os.path.basename(csv_files[0])

                data_frames = []
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        df['source_file'] = os.path.basename(csv_file)
                        data_frames.append(df)
                    except Exception:
                        pass
                if not data_frames: return False
                self.data = pd.concat(data_frames, ignore_index=True)
                print(f"合并后数据总形状: {self.data.shape}")
                print(f"使用文件名标识: {self.data_filename}")
                return True
            else:
                return False
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def extract_vibration_features(self):
        if self.data is None: return None
        target_cols = [ 'F_y', 'F_z']
        available_cols = [col for col in target_cols if col in self.data.columns]
        self.features = self.data[available_cols].copy()
        return self.features

    def split_into_samples(self, num_samples=1000, points_per_sample=26761, vib_cols=None, pad_mode='pad_last'):
        if self.data is None: raise ValueError("数据未加载")
        if vib_cols is None: vib_cols = ['F_y', 'F_z']
        arr = self.data[vib_cols].values
        n_rows, n_channels = arr.shape
        total_needed = int(num_samples) * int(points_per_sample)

        if n_rows >= total_needed:
            arr_used = arr[:total_needed]
        else:
            pad_len = total_needed - n_rows
            if pad_mode == 'pad_last':
                pad_row = arr[-1, :].reshape(1, -1) if n_rows > 0 else np.zeros((1, n_channels))
                pad = np.tile(pad_row, (pad_len, 1))
                arr_used = np.vstack([arr, pad])
            else:
                arr_used = np.tile(arr, ((total_needed // n_rows) + 1, 1))[:total_needed]

        self.samples = arr_used.reshape((int(num_samples), int(points_per_sample), n_channels))
        self.num_samples = int(num_samples)
        self.points_per_sample = int(points_per_sample)
        self.sample_channels = vib_cols
        print(f"拆分完成: samples.shape = {self.samples.shape}")
        return self.samples

    def extract_sample_aggregate_features(self):
        if self.samples is None: raise ValueError("请先调用 split_into_samples")
        num_samples, _, n_channels = self.samples.shape
        cols = self.sample_channels
        feature_names = []
        for ch in cols:
            feature_names.extend(
                [f"{ch}_mean", f"{ch}_std", f"{ch}_max", f"{ch}_min", f"{ch}_rms", f"{ch}_kurtosis", f"{ch}_skewness"])

        feature_list = []
        for i in tqdm(range(num_samples), desc="样本特征计算"):
            sample = self.samples[i]
            feats = []
            for c in range(n_channels):
                data = sample[:, c].astype(float)
                feats.extend([np.mean(data), np.std(data), np.max(data), np.min(data), np.sqrt(np.mean(data ** 2)),
                              stats.kurtosis(data), stats.skew(data)])
            feature_list.append(feats)

        self.sample_features = pd.DataFrame(feature_list, columns=feature_names)
        return self.sample_features


class GMMPredictor:
    def __init__(self, n_components=5, random_state=42):
        self.gmm = None
        self.scaler = StandardScaler()
        self.random_state = random_state

    def fit_gmm(self, X, n_components_range=None):
        if isinstance(X, pd.DataFrame): X = X.values
        X_scaled = self.scaler.fit_transform(X)
        if n_components_range is None: n_components_range = range(1, 11)
        best_gmm, best_bic = None, np.inf
        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=self.random_state).fit(X_scaled)
            bic = gmm.bic(X_scaled)
            if bic < best_bic:
                best_bic, best_gmm = bic, gmm
        self.gmm = best_gmm
        print(f"最优GMM组件数: {self.gmm.n_components}")
        return self.gmm

    def calculate_residuals(self, X):
        if self.gmm is None: raise ValueError("模型未训练")
        if isinstance(X, pd.DataFrame): X = X.values
        X_scaled = self.scaler.transform(X)
        return -self.gmm.score_samples(X_scaled)


class VisualizationTool:
    def plot_original_time_domain(self, data, max_points=10000):
        vib_columns = [col for col in data.columns if col.startswith('Vib_')]
        if not vib_columns: return None, None
        data_sampled = data.iloc[::max(1, len(data) // max_points)].copy()
        fig, axes = plt.subplots(len(vib_columns), 1, figsize=(15, 4 * len(vib_columns)))
        if len(vib_columns) == 1: axes = [axes]
        for i, col in enumerate(vib_columns):
            axes[i].plot(data_sampled.index, data_sampled[col].values, alpha=0.7)
            axes[i].set_title(f'{col} 时域信号')
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, axes

    def plot_health_indicator(self, time_index, hi, threshold, title):
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(time_index, hi, 'b-', linewidth=1.5, label='健康指标 (已平滑)')
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold})')
        anomalies = hi < threshold
        if anomalies.any():
            ax.scatter(time_index[anomalies], hi[anomalies], color='r', s=20, label='异常点')
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax

    def plot_gmm_components(self, gmm, features_scaled):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].scatter(features_scaled[:, 0], features_scaled[:, 1], alpha=0.3, s=10)
        axes[0].scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=100, label='GMM中心')
        axes[0].set_title('特征分布与GMM中心')
        axes[1].bar(range(1, gmm.n_components + 1), gmm.weights_)
        axes[1].set_title('组件权重')
        plt.tight_layout()
        return fig, axes

    def plot_residual_analysis(self, residuals, hi):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(residuals, alpha=0.7)
        axes[0].set_title('残差序列')
        axes[1].hist(residuals, bins=50, alpha=0.7)
        axes[1].set_title('残差分布')
        plt.tight_layout()
        return fig, axes


def main():
    # --- 配置区 ---
    file_path = r"E:\铣刀数据\2023_7_铣刀\训练数据"
    base_save_root = r"E:\铣刀数据\2023_7_铣刀\comparison of models\构建的HI\GMM—HI"

    # 确保根目录存在
    if not os.path.exists(base_save_root):
        os.makedirs(base_save_root)

    # num_samples = 1000 # 移除旧的固定样本数配置
    train_ratio = 0.1
    threshold = 0.3

    preprocessor = DataPreprocessor(file_path)
    gmm_predictor = GMMPredictor()
    hi_builder = HealthIndicatorBuilder(window_size=50)
    visualizer = VisualizationTool()

    print(">>> 步骤1: 加载并切分数据")
    if not preprocessor.load_data():
        print("错误: 无法加载数据");
        return

    # ========================================================================
    # 根据读取的CSV文件名创建子文件夹
    # ========================================================================
    csv_filename_no_ext = os.path.splitext(preprocessor.data_filename)[0]
    save_dir = os.path.join(base_save_root, csv_filename_no_ext)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建数据保存目录 (基于文件名): {save_dir}")
    else:
        print(f"数据保存目录已存在: {save_dir}")

    preprocessor.extract_vibration_features()

    # ========================================================================
    # 关键修改：每满 20000 个数据合成一个样本
    # ========================================================================
    points_per_sample = 20000
    total_data_points = len(preprocessor.data)

    # 计算可以分多少个完整样本
    num_samples = total_data_points // points_per_sample

    print(f"数据切分策略: 固定每样本 {points_per_sample} 个数据点")
    print(f"总数据量: {total_data_points} -> 计算得出样本数量: {num_samples}")

    if num_samples == 0:
        print("错误：数据量不足以构成一个样本（少于20000点）")
        return

    preprocessor.split_into_samples(num_samples=num_samples, points_per_sample=points_per_sample)
    # ========================================================================

    sample_features = preprocessor.extract_sample_aggregate_features()

    print("\n>>> 步骤2: 训练健康基准模型 (GMM)")
    train_size = int(len(sample_features) * train_ratio)
    # 确保至少有数据用于训练
    if train_size < 2:
        train_size = min(10, len(sample_features))

    train_features = sample_features.iloc[:train_size]
    gmm_predictor.fit_gmm(train_features, n_components_range=range(2, 8))

    print("\n>>> 步骤3: 计算健康指标 (带均值滤波)")
    residuals = gmm_predictor.calculate_residuals(sample_features)
    hi = hi_builder.build_hi_from_residuals(residuals, method='exponential')

    anomalies = hi_builder.detect_anomalies(hi, threshold)
    trend, trend_level = hi_builder.calculate_hi_trend(hi)

    print(f"异常点数量: {np.sum(anomalies)} / {len(hi)}")
    print(f"健康趋势评估: {trend_level}")

    print("\n>>> 步骤4: 可视化结果保存")
    time_idx = np.arange(len(hi))

    # 绘图逻辑 (注意：所有保存路径都已更新为 save_dir)
    fig0, _ = visualizer.plot_original_time_domain(preprocessor.data)
    if fig0: fig0.savefig(os.path.join(save_dir, '../original_signal.png'))

    fig_hi, _ = visualizer.plot_health_indicator(time_idx, hi, threshold, "GMM-HI (Smoothed)")
    fig_hi.savefig(os.path.join(save_dir, 'health_indicator_smoothed.png'))

    fig_res, _ = visualizer.plot_residual_analysis(residuals, hi)
    fig_res.savefig(os.path.join(save_dir, '../residual_analysis.png'))

    # --- 关键保存步骤：保存 HI 数据到指定路径 ---
    hi_results = pd.DataFrame({
        'Sample_Index': time_idx,
        'Health_Indicator': hi,
        'Raw_Residual': residuals,
        'Is_Anomaly': anomalies.astype(int)
    })

    output_path = os.path.join(save_dir, 'GMM_HI_Results.csv')
    hi_results.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"HI 值及完整信息已保存至: {output_path}")
    print("分析完成！")
    plt.show()


if __name__ == "__main__":
    main()
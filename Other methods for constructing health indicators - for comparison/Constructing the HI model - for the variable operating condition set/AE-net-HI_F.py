import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 引入 TensorFlow 用于构建自编码器
import tensorflow as pd_tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HealthIndicatorBuilder:
    def __init__(self, window_size=50):
        self.window_size = int(window_size)

    def build_hi_from_residuals(self, residuals, method='exponential'):
        """
        将残差映射为归一化健康指标。
        对于自编码器，残差是重构误差(MSE)，误差越小越健康(接近1)，误差越大越故障(接近0)。
        """
        residuals = np.asarray(residuals, dtype=float)
        if residuals.size == 0:
            return np.array([])

        # 1. 原始映射逻辑
        if method == 'exponential':
            # 对于AE的MSE误差，使用指数映射非常合适
            # 选取95分位点作为分母，或者直接用max，控制衰减速度
            denom = np.percentile(residuals, 95)
            if denom <= 0:
                denom = np.max(residuals) + 1e-10
            # 误差越大，HI越小
            hi_raw = np.exp(-residuals / denom * 2.0)  # *2.0 是为了让曲线对异常更敏感，可调整
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

        # --- 均值滤波处理 ---
        if self.window_size > 1:
            window = np.ones(self.window_size) / self.window_size
            hi_smoothed = np.convolve(hi_raw, window, mode='same')
            pad_size = self.window_size // 2
            hi_smoothed[:pad_size] = hi_raw[:pad_size]
            hi_smoothed[-pad_size:] = hi_raw[-pad_size:]
            hi_final = hi_smoothed
        else:
            hi_final = hi_raw

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
    """
    保持原有的数据读取和处理类不变，增加了 data_filename 属性
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data_filename = "dataset"  # 默认值
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
                    # 记录文件名
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

                # 按照文件名排序，确保顺序一致
                csv_files.sort()

                # 如果是文件夹读取，取第一个CSV文件名作为标识，或者您可以根据需求修改
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
        target_cols = ['F_x', 'F_y', 'F_z']
        available_cols = [col for col in target_cols if col in self.data.columns]
        self.features = self.data[available_cols].copy()
        return self.features

    def split_into_samples(self, num_samples=1000, points_per_sample=26761, vib_cols=None, pad_mode='pad_last'):
        if self.data is None: raise ValueError("数据未加载")
        if vib_cols is None: vib_cols = ['F_x', 'F_y', 'F_z']
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


class AutoencoderPredictor:
    """
    替换原有的 GMMPredictor。
    使用自编码器进行特征重构，通过重构误差(MSE)来表征异常程度。
    """

    def __init__(self, encoding_dim_ratio=0.5, random_state=42):
        self.model = None
        self.scaler = StandardScaler()
        self.random_state = random_state
        self.encoding_dim_ratio = encoding_dim_ratio

        # 设置随机种子以保证可复现性
        pd_tf.random.set_seed(random_state)
        np.random.seed(random_state)

    def _build_model(self, input_dim):
        """构建简单的全连接自编码器"""
        encoding_dim = int(input_dim * self.encoding_dim_ratio)
        if encoding_dim < 1: encoding_dim = 1

        # 编码器
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(int(input_dim * 0.75), activation='relu')(input_layer)
        encoded = Dense(encoding_dim, activation='relu')(encoded)

        # 解码器
        decoded = Dense(int(input_dim * 0.75), activation='relu')(encoded)
        decoded = Dense(input_dim, activation='linear')(decoded)  # 输出层线性激活以重构数值

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return autoencoder

    def fit(self, X_train, epochs=50, batch_size=32, verbose=0):
        """
        训练自编码器。
        注意：X_train 应该只包含健康状态的数据。
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values

        # 归一化处理
        X_scaled = self.scaler.fit_transform(X_train)

        input_dim = X_scaled.shape[1]
        self.model = self._build_model(input_dim)

        print(f"开始训练自编码器 (Input Dim: {input_dim})...")
        # 自编码器的目标是输入等于输出
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=verbose,
            validation_split=0.1
        )
        print("自编码器训练完成。")
        return history

    def calculate_residuals(self, X):
        """
        计算重构误差 (MSE) 作为残差。
        """
        if self.model is None:
            raise ValueError("模型未训练")

        if isinstance(X, pd.DataFrame):
            X = X.values

        # 使用之前 fit 的 scaler 进行转换
        X_scaled = self.scaler.transform(X)

        # 预测（重构）
        X_pred = self.model.predict(X_scaled, verbose=0)

        # 计算均方误差 (MSE)
        # axis=1 表示计算每个样本所有特征的平均误差
        mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

        return mse


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
        ax.plot(time_index, hi, 'b-', linewidth=1.5, label='健康指标 (Autoencoder)')
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold})')
        anomalies = hi < threshold
        if anomalies.any():
            ax.scatter(time_index[anomalies], hi[anomalies], color='r', s=20, label='异常点')
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax

    def plot_reconstruction_error(self, residuals):
        """新增：绘制重构误差图"""
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(residuals, color='purple', alpha=0.7, label='Reconstruction Error (MSE)')
        ax.set_title('自编码器重构误差 (MSE)')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        plt.tight_layout()
        return fig, ax


def main():
    # --- 配置区 ---
    file_path = r"E:\铣刀数据\2023_7_铣刀\训练数据"
    base_save_root = r"E:\铣刀数据\2023_7_铣刀\comparison of models\构建的HI\AE_HI"

    # 确保根目录存在
    if not os.path.exists(base_save_root):
        os.makedirs(base_save_root)

    # num_samples = 1000  # 已移除固定样本数
    train_ratio = 0.1  # 使用前10%的数据作为健康数据训练AE
    threshold = 0.3

    preprocessor = DataPreprocessor(file_path)
    # 替换为自编码器预测器
    ae_predictor = AutoencoderPredictor(encoding_dim_ratio=0.5)
    hi_builder = HealthIndicatorBuilder(window_size=50)
    visualizer = VisualizationTool()

    print(">>> 步骤1: 加载并切分数据")
    if not preprocessor.load_data():
        print("错误: 无法加载数据");
        return

    # ==========================================
    # 根据读取的 CSV 文件名创建保存文件夹
    # ==========================================
    csv_filename_no_ext = os.path.splitext(preprocessor.data_filename)[0]
    save_dir = os.path.join(base_save_root, csv_filename_no_ext)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建数据保存目录 (基于文件名): {save_dir}")
    else:
        print(f"数据保存目录已存在: {save_dir}")
    # ==========================================

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

    print("\n>>> 步骤2: 训练健康基准模型 (Autoencoder)")
    # 关键点：只使用前一部分（认为是健康的）数据来训练自编码器
    train_size = int(len(sample_features) * train_ratio)
    # 确保训练数据不为空
    if train_size < 2:
        train_size = min(10, len(sample_features))

    train_features = sample_features.iloc[:train_size]

    # 训练模型
    ae_predictor.fit(train_features, epochs=100, batch_size=16, verbose=1)

    print("\n>>> 步骤3: 计算健康指标 (基于重构误差)")
    # 对所有样本计算重构误差
    residuals = ae_predictor.calculate_residuals(sample_features)

    # 将重构误差映射为 0-1 的 HI
    hi = hi_builder.build_hi_from_residuals(residuals, method='exponential')

    anomalies = hi_builder.detect_anomalies(hi, threshold)
    trend, trend_level = hi_builder.calculate_hi_trend(hi)

    print(f"异常点数量: {np.sum(anomalies)} / {len(hi)}")
    print(f"健康趋势评估: {trend_level}")

    print("\n>>> 步骤4: 可视化结果保存")
    time_idx = np.arange(len(hi))

    # 绘图逻辑 - 注意现在使用 save_dir
    fig0, _ = visualizer.plot_original_time_domain(preprocessor.data)
    if fig0: fig0.savefig(os.path.join(save_dir, '../original_signal.png'))

    fig_hi, _ = visualizer.plot_health_indicator(time_idx, hi, threshold, "Autoencoder-HI (Smoothed)")
    fig_hi.savefig(os.path.join(save_dir, 'ae_health_indicator.png'))

    # 绘制重构误差图
    fig_res, _ = visualizer.plot_reconstruction_error(residuals)
    fig_res.savefig(os.path.join(save_dir, 'reconstruction_error.png'))

    # --- 关键保存步骤：保存 HI 数据到指定路径 ---
    hi_results = pd.DataFrame({
        'Sample_Index': time_idx,
        'Health_Indicator': hi,
        'Reconstruction_MSE': residuals,  # 保存 MSE 而不是 GMM 的 log-likelihood
        'Is_Anomaly': anomalies.astype(int)
    })

    output_path = os.path.join(save_dir, 'AE_HI_Results.csv')
    hi_results.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"HI 值及完整信息已保存至: {output_path}")
    print("分析完成！")
    plt.show()


if __name__ == "__main__":
    main()
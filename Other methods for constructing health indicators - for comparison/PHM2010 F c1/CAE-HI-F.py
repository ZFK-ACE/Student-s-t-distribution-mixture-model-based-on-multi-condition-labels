import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# 引入 TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, UpSampling1D, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 1. 数据预处理类 (保持不变)
# ==============================================================================
class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_filename = "dataset"
        self.data = None
        self.samples = None
        self.sample_channels = None

    def load_data(self):
        try:
            if os.path.isfile(self.file_path):
                if self.file_path.lower().endswith('.csv'):
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
                csv_files.sort()
                self.data_filename = os.path.basename(csv_files[0])
                data_frames = []
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
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

    def split_into_samples(self, num_samples, points_per_sample, vib_cols=None):
        if self.data is None: raise ValueError("数据未加载")
        if vib_cols is None: vib_cols = [ 'F_y', 'F_z']

        # 提取数据
        arr = self.data[vib_cols].values

        # 确保数据足够
        total_needed = num_samples * points_per_sample
        if len(arr) < total_needed:
            print(f"警告：数据不足。需要 {total_needed}，实际 {len(arr)}。将截断样本数。")
            num_samples = len(arr) // points_per_sample
            total_needed = num_samples * points_per_sample

        arr_used = arr[:total_needed]

        # 重塑为 (样本数, 时间步, 通道数)
        self.samples = arr_used.reshape((num_samples, points_per_sample, len(vib_cols)))
        self.sample_channels = vib_cols
        print(f"拆分完成: samples.shape = {self.samples.shape}")
        return self.samples


# ==============================================================================
# 2. 卷积自编码器 HI 构建器 (无监督)
# ==============================================================================
class CAE_HI_Builder:
    def __init__(self, input_shape):
        """
        input_shape: (time_steps, channels) 例如 (20000, 3)
        """
        self.input_shape = input_shape
        self.model = self._build_cae()
        # 使用 StandardScaler 而不是 MinMaxScaler，保留信号的波动幅度特征
        self.scaler = StandardScaler()

    def _build_cae(self):
        """
        构建卷积自编码器：编码器压缩特征，解码器还原信号
        """
        input_sig = Input(shape=self.input_shape)

        # --- 编码器 (Encoder) ---
        # 卷积层提取特征，池化层降低维度
        x = Conv1D(filters=16, kernel_size=64, strides=2, activation='relu', padding='same')(input_sig)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)  # 降采样

        x = Conv1D(filters=32, kernel_size=32, strides=2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)  # 再次降采样

        # --- 瓶颈层 (Bottleneck) ---
        # 这里是压缩后的特征表示

        # --- 解码器 (Decoder) ---
        # 上采样恢复维度，卷积层恢复细节
        x = UpSampling1D(size=2)(x)
        x = Conv1D(filters=32, kernel_size=32, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling1D(size=2)(x)

        # 注意：这里的 UpSampling 次数需要根据输入长度仔细调整，以确保输出维度与输入一致
        # 为了通用性，最后通常加一个 UpSampling 或 Conv1D 来匹配维度
        # 这里的网络结构针对 20000 点进行了简化，实际可能需要根据 shape 微调
        # 我们使用 Resize 策略或者 padding='same' 配合 UpSampling 尽量对齐
        # 为保证输出尺寸严格等于输入，通常最后用一层 Dense 或者 调整 UpSampling 策略
        # 这里为了演示稳健性，我们在最后使用 Upsampling 后可能尺寸不完全匹配，
        # Keras 现在的 Conv1DTranspose 更方便，但 UpSampling 兼容性更好。
        # 简单策略：如果维度不匹配，训练时会报错。
        # 针对 20000 点：
        # 20000 -> (stride2) 10000 -> (pool2) 5000 -> (stride2) 2500 -> (pool2) 1250
        # 1250 -> (up2) 2500 -> (up2) 5000 -> (up2) 10000 -> (up2) 20000

        x = UpSampling1D(size=2)(x)
        x = Conv1D(filters=16, kernel_size=64, activation='relu', padding='same')(x)
        x = UpSampling1D(size=2)(x)

        # 输出层：3个通道，线性激活（为了还原数值）
        decoded = Conv1D(filters=self.input_shape[1], kernel_size=3, activation='linear', padding='same')(x)

        model = Model(input_sig, decoded)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def fit_scaler(self, X_healthy):
        # 仅利用健康数据拟合标准化器
        n, t, c = X_healthy.shape
        reshaped = X_healthy.reshape(-1, c)
        self.scaler.fit(reshaped)

    def prepare_data(self, samples):
        n, t, c = samples.shape
        reshaped = samples.reshape(-1, c)
        scaled = self.scaler.transform(reshaped)
        return scaled.reshape(n, t, c)

    def train(self, X_train, epochs=50, batch_size=16):
        """
        训练自编码器
        X_train: 必须仅包含健康样本
        """
        print("开始训练卷积自编码器 (仅使用健康数据)...")
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # 自编码器的输入和标签都是 X_train
        history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1
        )
        return history

    def calculate_reconstruction_error(self, X):
        """
        计算重构误差 (MSE)
        """
        X_pred = self.model.predict(X, verbose=0)

        # 计算每个样本的均方误差 (Mean Squared Error)
        # 维度变换: (n, t, c) -> (n, )
        mse = np.mean(np.square(X - X_pred), axis=(1, 2))
        return mse


# ==============================================================================
# 3. 可视化工具
# ==============================================================================
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

    def plot_mse_trend(self, mse, title):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(mse, color='purple', alpha=0.7, label='Reconstruction Error (MSE)')
        ax.set_title(title)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        plt.tight_layout()
        return fig, ax

    def plot_final_hi(self, hi, title):
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(hi, 'b-', linewidth=1.5, label='Data-Driven HI')
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(title)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Health Indicator (0-1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax


# ==============================================================================
# 主程序
# ==============================================================================
def main():
    # --- 配置区 ---
    file_path = r"E:\铣刀数据\2023_7_铣刀\训练数据"
    base_save_root = r"E:\铣刀数据\2023_7_铣刀\comparison of models\构建的HI\CAE_HI"

    if not os.path.exists(base_save_root):
        os.makedirs(base_save_root)

    preprocessor = DataPreprocessor(file_path)
    visualizer = VisualizationTool()

    print(">>> 步骤1: 加载数据")
    if not preprocessor.load_data():
        print("错误: 无法加载数据")
        return

    csv_filename_no_ext = os.path.splitext(preprocessor.data_filename)[0]
    save_dir = os.path.join(base_save_root, csv_filename_no_ext)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 数据切分 ---
    points_per_sample = 20000
    total_data_points = len(preprocessor.data)
    num_samples = total_data_points // points_per_sample
    print(f">>> 步骤2: 数据切分 (每样本 {points_per_sample} 点), 共 {num_samples} 个样本")

    if num_samples < 10:
        print("错误：样本数量过少")
        return

    raw_samples = preprocessor.split_into_samples(num_samples, points_per_sample)

    print("\n>>> 步骤3: 卷积自编码器建模")
    input_shape = (raw_samples.shape[1], raw_samples.shape[2])
    cae_builder = CAE_HI_Builder(input_shape)

    # 1. 划分健康数据 (用于训练)
    # 假设前 15% 的数据是健康的
    train_ratio = 0.15
    train_size = max(2, int(num_samples * train_ratio))
    healthy_samples = raw_samples[:train_size]
    print(f"使用前 {train_size} 个样本作为健康基准进行训练")

    # 2. 数据标准化 (Fit on Healthy, Transform All)
    cae_builder.fit_scaler(healthy_samples)
    X_train = cae_builder.prepare_data(healthy_samples)
    X_all = cae_builder.prepare_data(raw_samples)

    # 3. 训练 (无监督：输入=输出)
    history = cae_builder.train(X_train, epochs=30, batch_size=8)

    print("\n>>> 步骤4: 计算重构误差并构建 HI")
    # 计算所有数据的重构误差
    # 误差越大 -> 越不像健康样本 -> 退化越严重
    mse_values = cae_builder.calculate_reconstruction_error(X_all)

    # --- 将 MSE 映射为 0-1 的 HI ---
    # 逻辑：MSE 小 (健康) -> HI 接近 1
    #       MSE 大 (故障) -> HI 接近 0
    # 公式：HI = 1 - (MSE - min) / (max - min)
    # 或者为了突出退化趋势，也可以用指数映射

    mse_min = np.min(mse_values)
    mse_max = np.max(mse_values)

    if mse_max == mse_min:
        hi = np.ones_like(mse_values)
    else:
        # 归一化反转
        hi = 1.0 - (mse_values - mse_min) / (mse_max - mse_min)

    # 平滑处理 (使趋势更清晰)
    window_size = 50
    if len(hi) > window_size:
        window = np.ones(window_size) / window_size
        hi_smoothed = np.convolve(hi, window, mode='same')
        hi_smoothed[:window_size // 2] = hi[:window_size // 2]
        hi_smoothed[-window_size // 2:] = hi[-window_size // 2:]
        hi_final = np.clip(hi_smoothed, 0.0, 1.0)
    else:
        hi_final = hi

    print("\n>>> 步骤5: 可视化与保存")
    time_idx = np.arange(len(hi_final))

    # 1. 原始信号
    fig0, _ = visualizer.plot_original_time_domain(preprocessor.data)
    if fig0: fig0.savefig(os.path.join(save_dir, 'original_signal.png'))

    # 2. MSE 原始趋势 (这是最真实的退化特征)
    fig_mse, _ = visualizer.plot_mse_trend(mse_values, "Raw Reconstruction Error (Degradation Trend)")
    fig_mse.savefig(os.path.join(save_dir, 'cae_mse_trend.png'))

    # 3. 最终 HI
    fig_hi, _ = visualizer.plot_final_hi(hi_final, "CNN-AE Data-Driven HI (0-1)")
    fig_hi.savefig(os.path.join(save_dir, 'cae_health_indicator.png'))

    # 4. 保存数据
    hi_results = pd.DataFrame({
        'Sample_Index': time_idx,
        'Health_Indicator': hi_final,
        'Reconstruction_MSE': mse_values
    })
    output_path = os.path.join(save_dir, 'CNN_AE_HI_Results.csv')
    hi_results.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"HI 值及完整数据已保存至: {output_path}")
    print("分析完成！")
    plt.show()


if __name__ == "__main__":
    main()
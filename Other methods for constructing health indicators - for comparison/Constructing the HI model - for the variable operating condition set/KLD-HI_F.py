import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HealthIndicatorBuilder:
    def __init__(self, window_size=50):
        self.window_size = int(window_size)

    def normalize_to_hi(self, kld_values):
        """
        将 KLD (越小越好) 转换为 HI (0-1, 越大越好)
        使用指数映射: HI = exp(-KLD / scale)
        """
        kld_values = np.asarray(kld_values, dtype=float)

        # 自动选择缩放因子：
        # 选取 KLD 的一个高分位点（如95%或最大值）作为分母，控制衰减速度
        # 这样可以确保故障时的 KLD 对应较低的 HI
        scale_factor = np.percentile(kld_values, 95)
        if scale_factor <= 1e-6:
            scale_factor = np.max(kld_values) + 1e-6

        # 计算 HI
        hi_raw = np.exp(-kld_values / scale_factor * 2.5)  # *2.5 是调节系数，控制曲线陡峭程度

        # 平滑处理
        if self.window_size > 1:
            window = np.ones(self.window_size) / self.window_size
            hi_smoothed = np.convolve(hi_raw, window, mode='same')
            # 边缘填充
            hi_smoothed[:self.window_size // 2] = hi_raw[:self.window_size // 2]
            hi_smoothed[-self.window_size // 2:] = hi_raw[-self.window_size // 2:]
            hi_final = hi_smoothed
        else:
            hi_final = hi_raw

        return np.clip(hi_final, 0.0, 1.0)

    def detect_anomalies(self, hi, threshold=0.3):
        return hi < threshold


class DataPreprocessor:
    """ 保持原有数据读取类不变 """

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
                    except:
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
        if vib_cols is None: vib_cols = ['F_x', 'F_y', 'F_z']
        arr = self.data[vib_cols].values
        total_needed = num_samples * points_per_sample
        if len(arr) < total_needed:
            num_samples = len(arr) // points_per_sample
            total_needed = num_samples * points_per_sample
        arr_used = arr[:total_needed]
        self.samples = arr_used.reshape((num_samples, points_per_sample, len(vib_cols)))
        self.sample_channels = vib_cols
        print(f"拆分完成: samples.shape = {self.samples.shape}")
        return self.samples


class KLDPredictor:
    """
    KLD 预测器：计算样本幅值分布与健康基准分布的差异
    """

    def __init__(self, num_bins=100):
        self.num_bins = num_bins
        self.reference_dist = None  # 健康基准分布 Q
        self.bin_edges = None  # 直方图的边界
        self.epsilon = 1e-10  # 防止除零的小数

    def fit(self, healthy_samples):
        """
        使用健康样本构建基准分布 Q
        healthy_samples shape: (N_healthy, Points, Channels)
        """
        # 将所有健康样本展平，统计整体的幅值分布
        # 这里为了简化，我们计算所有通道数据的综合分布，或者您可以只取单轴
        # 也可以计算欧几里得范数 (Composite Signal)
        flattened_data = np.linalg.norm(healthy_samples, axis=2).flatten()

        # 确定直方图的范围 (为了覆盖未来可能的故障数据，范围可以稍微放宽)
        data_min = np.min(flattened_data)
        data_max = np.max(flattened_data)
        # 扩宽范围以防故障数据超出边界
        range_min = data_min - (data_max - data_min) * 0.5
        range_max = data_max + (data_max - data_min) * 1.5

        # 计算基准直方图 (density=True 保证积分为1，即概率密度)
        hist, self.bin_edges = np.histogram(
            flattened_data, bins=self.num_bins, range=(range_min, range_max), density=True
        )

        # 添加 epsilon 并归一化，防止出现 0 概率
        hist = hist + self.epsilon
        self.reference_dist = hist / np.sum(hist)

        print("KLD 基准分布构建完成。")

    def calculate_kld(self, samples):
        """
        计算每个样本的 KLD 值
        samples shape: (N, Points, Channels)
        """
        if self.reference_dist is None:
            raise ValueError("请先调用 fit() 构建基准分布")

        kld_values = []

        # 对每个样本进行循环
        for i in tqdm(range(len(samples)), desc="计算 KLD"):
            # 计算该样本的幅值 (欧几里得范数，合并三轴)
            sample_data = np.linalg.norm(samples[i], axis=1)

            # 计算当前样本的直方图 P (使用与基准相同的 bin_edges)
            hist, _ = np.histogram(sample_data, bins=self.bin_edges, density=True)

            # 平滑处理
            hist = hist + self.epsilon
            current_dist = hist / np.sum(hist)

            # 计算 KLD (P || Q) = sum(P * log(P / Q))
            # scipy.stats.entropy(pk, qk) 计算的就是 KLD
            kld = stats.entropy(pk=current_dist, qk=self.reference_dist)
            kld_values.append(kld)

        return np.array(kld_values)


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

    def plot_kld_raw(self, kld_values):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(kld_values, color='purple', alpha=0.7, label='Raw KLD Value')
        ax.set_title('Kullback-Leibler Divergence (Distance from Healthy)')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('KLD (nats)')
        ax.legend()
        plt.tight_layout()
        return fig, ax

    def plot_hi(self, hi, title):
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(hi, 'b-', linewidth=1.5, label='KLD-Based HI')
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(title)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Health Indicator (0-1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax


def main():
    # --- 配置区 ---
    file_path = r"E:\铣刀数据\2023_7_铣刀\训练数据"
    base_save_root = r"E:\铣刀数据\2023_7_铣刀\comparison of models\构建的HI\KLD_HI"

    if not os.path.exists(base_save_root):
        os.makedirs(base_save_root)

    preprocessor = DataPreprocessor(file_path)
    kld_predictor = KLDPredictor(num_bins=200)  # 增加bins数量以捕捉细微分布变化
    hi_builder = HealthIndicatorBuilder(window_size=50)
    visualizer = VisualizationTool()

    print(">>> 步骤1: 加载并切分数据")
    if not preprocessor.load_data():
        print("错误: 无法加载数据")
        return

    # 创建保存目录
    csv_filename_no_ext = os.path.splitext(preprocessor.data_filename)[0]
    save_dir = os.path.join(base_save_root, csv_filename_no_ext)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 切分数据 (每样本 20000 点)
    points_per_sample = 20000
    total_data_points = len(preprocessor.data)
    num_samples = total_data_points // points_per_sample

    print(f"数据切分策略: 每样本 {points_per_sample} 个点")
    print(f"总数据量: {total_data_points} -> 样本数量: {num_samples}")

    if num_samples < 5:
        print("错误: 数据不足")
        return

    # 获取原始振动波形样本 (Samples, TimeSteps, Channels)
    samples = preprocessor.split_into_samples(num_samples=num_samples, points_per_sample=points_per_sample)

    print("\n>>> 步骤2: 训练 KLD 基准分布")
    # 使用前 10% 的样本作为健康基准
    train_ratio = 0.1
    train_size = max(2, int(num_samples * train_ratio))
    healthy_samples = samples[:train_size]

    # 拟合基准分布 (fit)
    kld_predictor.fit(healthy_samples)

    print("\n>>> 步骤3: 计算 KLD 并构建 HI")
    # 计算所有样本的 KLD
    kld_values = kld_predictor.calculate_kld(samples)

    # 将 KLD 映射为 0-1 的 HI
    hi = hi_builder.normalize_to_hi(kld_values)

    # 趋势分析
    slope, _ = np.polyfit(np.arange(len(hi)), hi, 1)
    trend_desc = "下降 (退化)" if slope < 0 else "上升 (异常)"
    print(f"HI 趋势斜率: {slope:.6f} -> {trend_desc}")

    print("\n>>> 步骤4: 可视化与保存")
    time_idx = np.arange(len(hi))

    # 1. 原始信号
    fig0, _ = visualizer.plot_original_time_domain(preprocessor.data)
    if fig0: fig0.savefig(os.path.join(save_dir, 'original_signal.png'))

    # 2. 原始 KLD 趋势 (验证物理意义: 应该随磨损变大)
    fig_kld, _ = visualizer.plot_kld_raw(kld_values)
    fig_kld.savefig(os.path.join(save_dir, 'raw_kld_trend.png'))

    # 3. 最终 HI
    fig_hi, _ = visualizer.plot_hi(hi, "KLD-Based Health Indicator (0-1)")
    fig_hi.savefig(os.path.join(save_dir, 'kld_health_indicator.png'))

    # 4. 保存数据
    hi_results = pd.DataFrame({
        'Sample_Index': time_idx,
        'Health_Indicator': hi,
        'Raw_KLD': kld_values
    })

    output_path = os.path.join(save_dir, 'KLD_HI_Results.csv')
    hi_results.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"结果已保存至: {output_path}")
    print("分析完成！")
    plt.show()


if __name__ == "__main__":
    main()
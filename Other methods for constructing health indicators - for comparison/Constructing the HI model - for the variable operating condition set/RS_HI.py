import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        if vib_cols is None: vib_cols = ['Vib_x', 'Vib_y', 'Vib_z']
        arr = self.data[vib_cols].values
        total_needed = num_samples * points_per_sample
        if len(arr) < total_needed:
            print(f"警告：数据不足。需要 {total_needed}，实际 {len(arr)}。将截断样本数。")
            num_samples = len(arr) // points_per_sample
            total_needed = num_samples * points_per_sample
        arr_used = arr[:total_needed]
        self.samples = arr_used.reshape((num_samples, points_per_sample, len(vib_cols)))
        self.sample_channels = vib_cols
        print(f"拆分完成: samples.shape = {self.samples.shape}")
        return self.samples


# ==============================================================================
# 2. RS (Rescaled Range) 分析预测器
# ==============================================================================
class RSPredictor:
    def __init__(self, min_chunk_size=50):
        self.min_chunk_size = min_chunk_size

    def calculate_hurst(self, time_series):
        """
        使用 R/S 分析法计算 Hurst 指数
        """
        time_series = np.array(time_series)
        N = len(time_series)

        # 定义不同的子区间长度 n
        # 从 min_chunk_size 到 N/2，生成对数分布的长度
        max_chunk_size = N // 2
        if max_chunk_size < self.min_chunk_size:
            # 如果样本太短，直接计算整体
            chunk_sizes = [N]
        else:
            # 生成约 20 个不同尺度的区间长度
            chunk_sizes = np.unique(np.logspace(
                np.log10(self.min_chunk_size),
                np.log10(max_chunk_size),
                num=15
            ).astype(int))
            # 过滤掉不合理的尺寸
            chunk_sizes = chunk_sizes[chunk_sizes > 10]

        rs_values = []
        n_values = []

        for n in chunk_sizes:
            # 将序列切分为多个长度为 n 的子区间
            num_chunks = N // n
            if num_chunks < 1: continue

            # 截取可以整除的部分
            reshaped_series = time_series[:num_chunks * n].reshape(num_chunks, n)

            # 对每个子区间计算 R/S
            # 1. 均值
            means = np.mean(reshaped_series, axis=1, keepdims=True)
            # 2. 离差序列 (Y)
            y = reshaped_series - means
            # 3. 累积离差序列 (Z)
            z = np.cumsum(y, axis=1)
            # 4. 极差 (Range, R) = max(Z) - min(Z)
            r = np.max(z, axis=1) - np.min(z, axis=1)
            # 5. 标准差 (S)
            s = np.std(reshaped_series, axis=1, ddof=1)

            # 处理标准差为0的情况
            s = np.where(s == 0, 1e-10, s)

            # 6. 重标极差 (R/S)
            rs = r / s

            # 计算当前尺度 n 下的平均 R/S
            avg_rs = np.mean(rs)

            rs_values.append(avg_rs)
            n_values.append(n)

        # 线性拟合 log(R/S) vs log(n)
        # 斜率即为 Hurst 指数
        if len(n_values) < 2:
            return 0.5  # 默认随机游走

        log_n = np.log(n_values)
        log_rs = np.log(rs_values)

        slope, intercept = np.polyfit(log_n, log_rs, 1)
        return slope

    def process_samples(self, samples):
        """
        计算所有样本的综合 Hurst 指数
        samples: (num_samples, points_per_sample, num_channels)
        """
        hurst_list = []

        for i in tqdm(range(len(samples)), desc="计算 RS (Hurst)"):
            sample = samples[i]
            # 计算该样本的综合信号（向量模长），将三轴合并为单轴
            # 也可以分别计算三轴再求平均
            composite_signal = np.linalg.norm(sample, axis=1)

            # 计算 Hurst
            h = self.calculate_hurst(composite_signal)
            hurst_list.append(h)

        return np.array(hurst_list)


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

    def plot_hurst_trend(self, hurst_values):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(hurst_values, color='teal', alpha=0.8, label='Hurst Exponent (from R/S)')
        ax.set_title('Fractal Characteristic Trend (Hurst Exponent)')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Hurst Index (0-1)')
        ax.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='Random Walk (H=0.5)')
        ax.legend()
        plt.tight_layout()
        return fig, ax

    def plot_hi(self, hi, title):
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(hi, 'b-', linewidth=1.5, label='RS-Based HI')
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
    base_save_root = r"E:\铣刀数据\2023_7_铣刀\comparison of models\构建的HI\RS_HI"

    if not os.path.exists(base_save_root):
        os.makedirs(base_save_root)

    preprocessor = DataPreprocessor(file_path)
    rs_predictor = RSPredictor()
    visualizer = VisualizationTool()

    print(">>> 步骤1: 加载并切分数据")
    if not preprocessor.load_data():
        print("错误: 无法加载数据")
        return

    csv_filename_no_ext = os.path.splitext(preprocessor.data_filename)[0]
    save_dir = os.path.join(base_save_root, csv_filename_no_ext)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 数据切分策略 ---
    points_per_sample = 20000
    total_data_points = len(preprocessor.data)
    num_samples = total_data_points // points_per_sample
    print(f"数据切分策略: 每样本 {points_per_sample} 个点")
    print(f"总数据量: {total_data_points} -> 样本数量: {num_samples}")

    if num_samples < 5:
        print("错误：数据量不足")
        return

    samples = preprocessor.split_into_samples(num_samples, points_per_sample)

    print("\n>>> 步骤2: 计算 Hurst 指数 (R/S 分析)")
    # R/S 分析计算量较大，tqdm 会显示进度
    hurst_values = rs_predictor.process_samples(samples)

    print("\n>>> 步骤3: 构建健康指标 (HI)")
    # --- HI 构建逻辑 ---
    # Hurst 指数通常在 0 到 1 之间。
    # 物理意义：
    # H = 0.5: 随机游走 (无记忆)
    # H > 0.5: 持久性 (趋势增强)，通常健康状态下机械系统的振动具有较强的规律性(Persistence)
    # H < 0.5: 反持久性 (均值回归)
    # 假设：随着磨损，系统非线性增加，信号可能会变得更无序，导致 H 发生变化（通常是下降或波动）。

    # 我们采用自适应归一化：
    # 假设第一个样本是健康的 (基准)
    h_baseline = np.mean(hurst_values[:5])

    # 定义 HI：衡量当前 H 与健康基准的“相似度”或“距离”
    # 方法1 (距离法): HI = 1 - |H - H_base| / max_dist
    # 方法2 (直接归一化): 视 H 的变化趋势而定。

    # 这里使用通用的 Min-Max 归一化，方向取决于趋势
    # 先做简单的平滑以便判断趋势
    w = 20
    h_smooth = np.convolve(hurst_values, np.ones(w) / w, mode='same')

    # 自动判断退化方向：如果整体趋势是下降，则按正向归一化；如果是上升，则反向
    slope = np.polyfit(np.arange(len(h_smooth)), h_smooth, 1)[0]

    h_min = np.min(hurst_values)
    h_max = np.max(hurst_values)

    if slope < 0:
        # H 下降代表退化：HI = (H - min) / (max - min)
        print("检测到 Hurst 指数呈下降趋势 (符合混沌度增加)")
        hi = (hurst_values - h_min) / (h_max - h_min)
    else:
        # H 上升代表退化：HI = (max - H) / (max - min)
        print("检测到 Hurst 指数呈上升趋势")
        hi = (h_max - hurst_values) / (h_max - h_min)

    # 再次平滑 HI 曲线
    window_size = 50
    if len(hi) > window_size:
        window = np.ones(window_size) / window_size
        hi_smoothed = np.convolve(hi, window, mode='same')
        hi_smoothed[:window_size // 2] = hi[:window_size // 2]
        hi_smoothed[-window_size // 2:] = hi[-window_size // 2:]
        hi_final = np.clip(hi_smoothed, 0.0, 1.0)
    else:
        hi_final = hi

    print("\n>>> 步骤4: 可视化与保存")
    time_idx = np.arange(len(hi_final))

    # 1. 原始信号
    fig0, _ = visualizer.plot_original_time_domain(preprocessor.data)
    if fig0: fig0.savefig(os.path.join(save_dir, 'original_signal.png'))

    # 2. Hurst 原始趋势
    fig_h, _ = visualizer.plot_hurst_trend(hurst_values)
    fig_h.savefig(os.path.join(save_dir, 'hurst_trend.png'))

    # 3. 最终 HI
    fig_hi, _ = visualizer.plot_hi(hi_final, "RS Analysis (Hurst) Based HI")
    fig_hi.savefig(os.path.join(save_dir, 'rs_health_indicator.png'))

    # 4. 保存数据
    hi_results = pd.DataFrame({
        'Sample_Index': time_idx,
        'Health_Indicator': hi_final,
        'Hurst_Exponent': hurst_values
    })
    output_path = os.path.join(save_dir, 'RS_HI_Results.csv')
    hi_results.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"HI 值及完整数据已保存至: {output_path}")
    print("分析完成！")
    plt.show()


if __name__ == "__main__":
    main()
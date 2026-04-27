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


# ==============================================================================
# 数据预处理类 (保持原文件内容不变)
# ==============================================================================
class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_filename = "dataset"
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


# ==============================================================================
# 可视化工具类
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

    def plot_health_indicator(self, time_index, hi, title):
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(time_index, hi, 'b-', linewidth=1.5, label='RMS-Based HI')
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(title)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Health Indicator (0-1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax

    def plot_rms_trend(self, rms_values):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(rms_values, color='orange', alpha=0.8, label='Composite RMS')
        ax.set_title('Composite RMS Trend (Raw)')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('RMS Amplitude')
        ax.legend()
        plt.tight_layout()
        return fig, ax


# ==============================================================================
# 主程序
# ==============================================================================
def main():
    # --- 配置区 ---
    file_path = r"E:\铣刀数据\2023_7_铣刀\训练数据"
    base_save_root = r"E:\铣刀数据\2023_7_铣刀\comparison of models\构建的HI\RMS_HI"

    # 确保根目录存在
    if not os.path.exists(base_save_root):
        os.makedirs(base_save_root)

    preprocessor = DataPreprocessor(file_path)
    visualizer = VisualizationTool()

    print(">>> 步骤1: 加载并切分数据")
    if not preprocessor.load_data():
        print("错误: 无法加载数据")
        return

    # 根据读取的CSV文件名创建保存文件夹
    csv_filename_no_ext = os.path.splitext(preprocessor.data_filename)[0]
    save_dir = os.path.join(base_save_root, csv_filename_no_ext)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建数据保存目录 (基于文件名): {save_dir}")
    else:
        print(f"数据保存目录已存在: {save_dir}")

    # 特征提取
    preprocessor.extract_vibration_features()

    # ==========================================================================
    # 修改处：不再使用固定的num_samples，而是固定每20000个点为一个样本
    # ==========================================================================
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
    # ==========================================================================

    sample_features = preprocessor.extract_sample_aggregate_features()

    print("\n>>> 步骤2: 基于 RMS 构建健康指标 (HI)")

    # 1. 获取 RMS 列 (DataPreprocessor 自动生成了 _rms 后缀的特征)
    rms_cols = [c for c in sample_features.columns if c.endswith('_rms')]
    if not rms_cols:
        print("错误: 未找到 RMS 特征列")
        return

    print(f"使用 RMS 特征列: {rms_cols}")

    # 2. 计算综合 RMS (欧几里得范数，反映整体振动能量)
    # RMS_composite = sqrt(RMS_x^2 + RMS_y^2 + RMS_z^2)
    rms_values = np.sqrt(np.sum(sample_features[rms_cols].values ** 2, axis=1))

    # 3. 归一化构建 HI (映射到 0-1)
    # 逻辑：健康时 RMS 小 (HI -> 1)，故障时 RMS 大 (HI -> 0)
    # 公式：HI = (RMS_max - RMS_current) / (RMS_max - RMS_min)
    rms_min = np.min(rms_values)
    rms_max = np.max(rms_values)

    if rms_max == rms_min:
        hi = np.ones_like(rms_values)
    else:
        hi = (rms_max - rms_values) / (rms_max - rms_min)

    # 4. 简单的均值平滑 (使曲线更平滑，便于观察趋势)
    window_size = 50
    if len(hi) > window_size:
        window = np.ones(window_size) / window_size
        hi_smoothed = np.convolve(hi, window, mode='same')
        # 边缘填充
        hi_smoothed[:window_size // 2] = hi[:window_size // 2]
        hi_smoothed[-window_size // 2:] = hi[-window_size // 2:]
        # 确保范围在 0-1
        hi = np.clip(hi_smoothed, 0.0, 1.0)

    print("\n>>> 步骤3: 可视化结果保存")
    time_idx = np.arange(len(hi))

    # 绘制并保存原始信号图
    fig0, _ = visualizer.plot_original_time_domain(preprocessor.data)
    if fig0: fig0.savefig(os.path.join(save_dir, '../original_signal.png'))

    # 绘制并保存 RMS 趋势图
    fig_rms, _ = visualizer.plot_rms_trend(rms_values)
    fig_rms.savefig(os.path.join(save_dir, 'rms_trend_raw.png'))

    # 绘制并保存最终 HI 图
    fig_hi, _ = visualizer.plot_health_indicator(time_idx, hi, "RMS-Based Health Indicator (0-1)")
    fig_hi.savefig(os.path.join(save_dir, 'rms_health_indicator.png'))

    # 保存 CSV 数据
    hi_results = pd.DataFrame({
        'Sample_Index': time_idx,
        'Health_Indicator': hi,
        'Composite_RMS': rms_values
    })

    output_path = os.path.join(save_dir, 'RMS_HI_Results.csv')
    hi_results.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"HI 值及完整数据已保存至: {output_path}")
    print("分析完成！")
    plt.show()


if __name__ == "__main__":
    main()
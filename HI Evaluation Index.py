import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr

def calculate_hi_metrics(df, hi_columns, alpha=50):
    """
    计算健康指标(HI)的各项评价指标
    :param df: 数据框
    :param hi_columns: 需要计算的健康指标列名列表
    :param alpha: 鲁棒性惩罚因子 (默认设为 10)，用于放大不同模型残差的差距
    """
    evaluation_results = []

    for col in hi_columns:
        if col not in df.columns:
            continue

        hi_val = df[col].values
        n = len(hi_val)

        # --- 1. 单调性 (Mon) ---
        diff = np.diff(hi_val)
        mon = abs(np.sum(diff > 0) - np.sum(diff < 0)) / (n - 1)

        # --- 2. 鲁棒性 (Rob) ---
        # 使用移动平均(窗口为50)提取趋势项
        trend = pd.Series(hi_val).rolling(window=50, center=True).mean().interpolate(limit_direction='both').values
        residual = hi_val - trend
        # 在这里引入了惩罚因子 alpha，乘以相对残差的绝对值
        rob = np.mean(np.exp(-alpha * np.abs(residual / (hi_val + 1e-6))))

        # --- 3. 趋势性 (Tre) ---
        # 严格按照公式 corr(HI_t, linspace(0, 1, t)) 计算
        trend_line = np.linspace(0, 1, n)
        trendability, _ = pearsonr(hi_val, trend_line)
        trendability = abs(trendability)

        # --- 4. 综合评价指标 (CI) ---
        # CI = 0.4 * Mon + 0.4 * Tre + 0.2 * Rob
        score = 0.4 * mon + 0.4 * trendability + 0.2 * rob

        # 提取模型名称用于行标签 (去掉 '-HI' 后缀)
        model_name = col.replace('-HI', '')

        # 按照需求整理列标签名称
        evaluation_results.append({
            'Model': model_name,
            'Mon': round(mon, 4),
            'CI': round(score, 4),
            'Tre': round(trendability, 4),
            'Rob': round(rob, 4)
        })

    return pd.DataFrame(evaluation_results)


# --- 主程序 ---
# 读取路径与保存路径
input_folder = r'E:\铣刀数据\2023_7_铣刀\comparison of models\evaluation index'
output_folder = r'E:\铣刀数据\2023_7_铣刀\comparison of models\index'

# 如果输出文件夹不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义目标 HI 列名
target_cols = ['AE-HI', 'CAE-HI', 'GMM-HI', 'KLD-HI', 'RMS-HI', 'MCt-HI','SMM','RT']

# 定义惩罚因子大小
penalty_alpha = 10

# 定义要求的行标签和列标签排序顺序
expected_rows = ['MCt', 'KLD', 'RMS', 'GMM', 'CAE', 'AE','SMM','RT']
expected_cols = ['Mon', 'CI', 'Tre', 'Rob']

try:
    # 寻找输入路径下的所有 CSV 文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    if not csv_files:
        print(f"在路径 {input_folder} 下未找到 CSV 文件。")
    else:
        for file_name in csv_files:
            file_path = os.path.join(input_folder, file_name)
            print(f"正在处理文件: {file_path}")

            # 读取数据
            data = pd.read_csv(file_path)

            # 执行计算
            results_df = calculate_hi_metrics(data, target_cols, alpha=penalty_alpha)

            if not results_df.empty:
                # 将 Model 列设为索引，使其成为行标签
                results_df.set_index('Model', inplace=True)

                # 提取存在的有效行，并按照指定的 expected_rows 顺序重排；列按 expected_cols 顺序重排
                valid_rows = [r for r in expected_rows if r in results_df.index]
                results_df = results_df.reindex(index=valid_rows, columns=expected_cols)

            # 构造保存文件名：原文件名 + _Evaluation.csv
            output_file_name = os.path.splitext(file_name)[0] + "_Evaluation.csv"
            output_path = os.path.join(output_folder, output_file_name)

            # 保存结果 (index=True 保留行标签)
            results_df.to_csv(output_path, index=True)
            print(f"评价结果已保存至: {output_path}")

        print("\n所有文件处理完毕。")

except Exception as e:
    print(f"运行出错: {e}")
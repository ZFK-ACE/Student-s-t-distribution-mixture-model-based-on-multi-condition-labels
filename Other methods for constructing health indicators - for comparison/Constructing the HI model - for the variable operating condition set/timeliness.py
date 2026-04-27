"""
健康指标构建模型及时性评测脚本
功能：
- 遍历当前目录下的模型脚本（以 `HI` 关键字识别，排除自身）
- 使用子进程运行每个脚本，监控执行时间及峰值内存
- 捕获子进程输出，提取模型参数大小
- 将结果保存为 CSV 表格，并在控制台打印
"""

import os
import sys
import time
import subprocess
import psutil
import threading
import pandas as pd
import re

# ==================== 配置区域 ====================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(MODEL_DIR, "../timeliness_results.csv")
# ==================================================

model_files = []
for f in os.listdir(MODEL_DIR):
    if f.endswith(".py") and "HI" in f and f != os.path.basename(__file__):
        model_files.append(os.path.join(MODEL_DIR, f))

print(f"发现 {len(model_files)} 个模型脚本：")
for m in model_files:
    print(f"  {os.path.basename(m)}")

results = []

# 【修复】内存监控线程函数
def monitor_memory(popen_proc, max_mem_list):
    max_mem = 0
    try:
        # 获取系统级进程对象
        p = psutil.Process(popen_proc.pid)
        # 使用 popen_proc.poll() 判断子进程是否结束
        while popen_proc.poll() is None:
            try:
                mem = p.memory_info().rss   # 实际物理内存（字节）
                if mem > max_mem:
                    max_mem = mem
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.1)   # 采样间隔 100ms
    except psutil.NoSuchProcess:
        pass
    max_mem_list.append(max_mem)

for script_path in model_files:
    script_name = os.path.basename(script_path)
    print(f"\n正在运行: {script_name} ...")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    start_time = time.time()

    proc = subprocess.Popen(
        [sys.executable, script_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors='ignore'
    )

    max_mem_list = []
    # 【修复】这里传入原生的 proc 对象，而不是 psutil.Process 对象
    monitor_thread = threading.Thread(target=monitor_memory, args=(proc, max_mem_list))
    monitor_thread.start()

    stdout_data, _ = proc.communicate()
    end_time = time.time()

    monitor_thread.join()

    if stdout_data:
        print(stdout_data.strip())

    peak_memory_mb = max_mem_list[0] / (1024 * 1024) if max_mem_list else 0
    elapsed_time = end_time - start_time

    retcode = proc.returncode
    if retcode != 0:
        print(f"  警告：脚本 {script_name} 返回非零退出码 {retcode}，可能执行失败。")

    # 提取参数大小
    param_size = "N/A"
    match = re.search(r"PARAM_SIZE:\s*([\d\.]+)", stdout_data)
    if match:
        param_size = float(match.group(1))

    results.append({
        "Script": script_name,
        "Total Time (s)": round(elapsed_time, 2),
        "Peak Memory (MB)": round(peak_memory_mb, 2),
        "Parameter Size": param_size,
        "Exit Code": retcode
    })

df = pd.DataFrame(results)
print("\n" + "="*75)
print("及时性评估结果汇总")
print("="*75)
print(df.to_string(index=False))

df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n结果已保存至: {OUTPUT_CSV}")
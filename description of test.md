# MCt: Health Indicator Construction Based on a Student's t-Distribution Mixture Model with Multi-Condition Labels

## 1. Project Overview

The core method first identifies the machining condition using current or acoustic-emission signals and then establishes an independent health baseline model for each condition. This reduces the possibility that changes in spindle speed, feed rate, cutting depth, and other operating conditions are misinterpreted as equipment degradation.

The project includes the complete method, an adapted version for the publicly available PHM2010 dataset, two ablation experiments, and a demonstration video of an actual experiment. The program can export condition labels, health indicators, and relevant intermediate results to facilitate figure generation and comparative analysis for research papers.

## 2. Project Structure

```text
Student-s-t-distribution-mixture-model-based-on-multi-condition-labels/
├── MCt/
│   ├── MCt.py
│   └── MCt-PHM2010.py
├── Experimental data acquisition/
│   └── Ablation experiment/
│       ├── Ablation experiment - Remove HI fusion - Trend HI.py
│       └── Ablation Experiment - Condition Division and DOffset.py
├── Attachment Description.md
├── Video Attachment 1.mp4
└── Video thumbnail image.png
```

The purpose of each file is as follows:

| File | Purpose |
|---|---|
| `MCt/MCt.py` | Main method: divides operating conditions using current signals and constructs and fuses the HI using cutting-force features |
| `MCt/MCt-PHM2010.py` | Version adapted to the publicly available PHM2010 dataset; clusters using AE signals and constructs the HI using force signals |
| `Ablation experiment - Remove HI fusion - Trend HI.py` | Removes HI fusion and retains only the multi-condition real-time HI and the median-filtered trend HI |
| `Ablation Experiment - Condition Division and DOffset.py` | Removes condition division and DOffset and uses a single global health model for comparison |
| `Video Attachment 1.mp4` | Demonstration of the method in an actual experiment; approximately 31.74 s, 1280×720, 30 fps |
| `Video thumbnail image.png` | Thumbnail of the demonstration video, 1280×720 |
| `Attachment Description.md` | Original brief attachment description |

## 3. Runtime Environment

Python 3.9 or later is recommended. The program depends on:

```text
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
openpyxl        # required for reading .xlsx files
xlrd            # may be required for reading legacy .xls files
```

Install the dependencies with:

```powershell
python -m pip install numpy pandas matplotlib seaborn scipy scikit-learn openpyxl xlrd
```

The current project does not provide a `requirements.txt` file or packaging configuration; dependencies must be installed manually.

## 4. Input Data Requirements for the Main Program

### 4.1 Supported File Formats

`MCt.py` supports:

- `.csv`
- `.xlsx`
- `.xls`
- `.txt` (tab-delimited)

When `data_path` points to a directory, the program searches only its top level and selects the largest file among those with supported formats as the input. It does not search recursively through subdirectories.

For CSV files, the following encodings are tried in sequence:

1. UTF-8;
2. GBK;
3. GB2312.

### 4.2 Channels Required by the Main Method

The main program requires at least one current channel and one cutting-force channel.

Current channels are used for condition clustering. Column names are identified automatically using the following keywords:

| Canonical column name | Recognized column-name fragments |
|---|---|
| `Cur_u` | `cur_u`, `current_u`, `i_u` |
| `Cur_v` | `cur_v`, `current_v`, `i_v` |
| `Cur_w` | `cur_w`, `current_w`, `i_w` |

Cutting-force channels are used to construct the HI:

| Canonical column name | Recognized column-name fragments |
|---|---|
| `F_x` | `f_x`, `force_x` |
| `F_y` | `f_y`, `force_y` |
| `F_z` | `f_z`, `force_z` |

Matching is case-insensitive. If multiple candidate columns correspond to the same channel, the first matching column in the data table is used.

Example of a minimal CSV header:

```csv
Cur_u,Cur_v,Cur_w,F_x,F_y,F_z
1.25,1.30,1.18,102.1,56.2,210.5
1.28,1.29,1.21,103.0,55.8,212.0
```

### 4.3 Data Length and Segmentation

The default window length is 20,000 raw samples:

```python
df_cluster_feats, df_hi_feats = split_and_featurize(df, chunk_size=20000)
```

If the length of the raw data is `N`, the final number of feature samples is:

```text
floor(N / 20000)
```

## 5. Main Program Outputs

### 5.1 Output Directories

The paths in the main program are:

```python
data_path = r"E:\铣刀数据\2023_7_铣刀\训练数据"
base_save_path = r"E:\铣刀数据\2023_7_铣刀\comparison of models\构建的HI\MY—HI"
```

Before running the program, change these values to valid local paths. If the input is a file, or a file successfully selected from a directory, the final output directory is:

```text
base_save_path/input-file-name-without-extension/
```

### 5.2 `Origin_Clustering_Data.csv`

Stores the features used for condition clustering:

| Field | Meaning |
|---|---|
| `Cur_u/Cur_v/Cur_w` | Current RMS for each window; the actual number of columns depends on the channels identified |
| `Labels_Filtered` | Condition labels after short-segment filtering |
| `PCA_1` | First principal-component coordinate of the clustering features |
| `PCA_2` | Second principal-component coordinate of the clustering features |

When only one current channel is identified, PCA has only one component, whereas the current saving code still accesses `pca_data[:,1]`. Therefore, at least two current channels are recommended for actual execution of the main program.

### 5.3 `Origin_HI_Fusion_Results.csv`

| Field | Meaning |
|---|---|
| `Sample_Index` | Index of the segmented feature sample |
| `Kalman_Filtered_HI` | Real-time HI |
| `Log_Likelihood` | Student's t mixture log-likelihood of the current sample |
| `Threshold` | Healthy-likelihood threshold for the current condition |
| `Cumulative_HI` | Cumulative anomaly HI |
| `Anomaly_Flag` | Anomaly flag: 1 for anomalous and 0 for normal |
| `Fused_HI_Raw` | Fused HI before median filtering |
| `Fused_HI_Median_Filtered` | Final fused HI after median filtering |
| `Weight_Kalman` | Weight of the real-time HI in the fusion |
| `Weight_Cumulative` | Weight of the cumulative HI in the fusion |
| `Cluster_Label` | Condition label of the current sample |

### 5.4 Visualization Results

`MCt.py` displays the following plots:

- Time series of current RMS values with cluster coloring;
- Clustering in the original feature space;
- PCA projections and the distribution of the number of conditions;
- HI, log-likelihood, HI distribution, and condition labels;
- Independent real-time HI;
- HI with condition boundaries;
- Cumulative anomaly HI;
- Comparison of the real-time HI, cumulative HI, and fused HI;
- Fusion confidence;
- Changes in fusion weights;
- Comparison of the fused HI before and after median filtering.

The main-program plots are displayed interactively using `plt.show()` and are not saved as image files by default. The program may pause at each plot window; execution resumes after the current window is closed.

## 6. How to Run

### 6.1 Modify the Paths

Open `MCt/MCt.py` and modify the following variables in `main()`:

```python
data_path = r"your input file or data directory"
base_save_path = r"your result output directory"
```

### 6.2 Run the Main Method

Open PowerShell in the project root directory and run:

```powershell
python ".\MCt\MCt.py"
```

### 6.3 Run the PHM2010 Version

```powershell
python ".\MCt\MCt-PHM2010.py"
```

### 6.4 Run the Ablation Experiments

Remove HI fusion and output only the real-time HI:

```powershell
python ".\Experimental data acquisition\Ablation experiment\Ablation experiment - Remove HI fusion - Trend HI.py"
```

Remove condition division and DOffset:

```powershell
python ".\Experimental data acquisition\Ablation experiment\Ablation Experiment - Condition Division and DOffset.py"
```

## 7. Notes on the PHM2010 Version

`MCt-PHM2010.py` retains the overall strategy of condition division, condition-specific HI modeling, and fusion, but accepts different signal types from the main program.

### 7.1 Input Channels

The program searches for the first acoustic-emission channel whose column name contains `AE` and copies it to:

```text
AE_1, AE_2, AE_3
```

The data in these three channels are identical and are used only to adapt the subsequent three-dimensional clustering and plotting structure. Force channels are still matched using `F_x/F_y/F_z` or `force_x/force_y/force_z`.

## 8. Notes on the Ablation Experiments

### 8.1 Removing HI Fusion: Trend HI

`Ablation experiment - Remove HI fusion - Trend HI.py` retains:

- Condition division based on current RMS;
- Automatic cluster selection using BIC;
- Condition-specific Student's t mixture health models;
- DOffset concatenation;
- Real-time HI.

Outputs include:

- `Origin_Clustering_Data.csv`;
- `Origin_Realtime_Trend_HI_Results.csv`;
- `Realtime_HI_Median_Filtered.png` (300 dpi).

The key ablation-result fields in the CSV file are:

```text
Kalman_Filtered_HI
Realtime_HI_Median_Filtered
```

This experiment can be used to evaluate the contribution of decision-level fusion relative to a single trend HI.

### 8.2 Removing Condition Division and DOffset

`Ablation Experiment - Condition Division and DOffset.py` does not use current-based clustering or establish independent models for different conditions. It uses only force signals, takes the first 50 feature samples as a unified healthy baseline, and applies a single global model to the entire dataset.

Its main characteristics are:

- Only force channels are required as input;
- All samples share one health model;
- No DOffset concatenation is performed;
- All `Cluster_Label` values are 0;
- The main workflow does not output cumulative HI or fused HI.

Outputs include:

- `Origin_HI_Results.csv`;
- `Realtime_HI_Median_Filtered.png` (300 dpi).

This experiment can be used to evaluate the role of condition division and DOffset in suppressing interference caused by condition switching.

## 9. Recommendations for Interpreting the Results

### 9.1 HI Values

- Near 1: the sample is similar to the healthy baseline for the current condition;
- Approximately 0.7–1: generally a healthy or mildly fluctuating stage;
- Approximately 0.3–0.7: gradual degradation or a mismatch with the condition model may be present;
- Below 0.3: one of the anomaly criteria implemented in the code is triggered;
- Near 0: the sample deviates substantially from the health model.

The HI is neither a failure probability nor a remaining useful life estimate. It is a relative health score derived from the current model and training baseline.

### 9.2 Condition Labels

The clustering labels `0,1,2,...` are merely unsupervised class identifiers and have no inherent meaning in terms of magnitude, health status, or machining-parameter order. The label numbers may be permuted across datasets or after refitting.

### 9.3 Anomaly Flags

`Anomaly_Flag=1` indicates that the sample triggered a statistical anomaly condition implemented in the code; it does not mean that a failure has been definitively confirmed. Interpretation should be combined with tool-wear measurements, machining quality, experimental time, and actual operating-condition records.

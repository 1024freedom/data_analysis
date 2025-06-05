# 财务造假检测数据预处理系统

## 项目概述

本项目是一个用于财务造假检测的数据预处理系统，专为审计和金融风控场景设计。系统提供从原始财务数据加载到特征工程的全流程处理，输出高质量的特征数据供后续建模使用。

## 主要功能

1. **数据加载与整合**：
   - 加载行业分类、财务数据和字段含义数据
   - 合并多源数据并筛选制造业企业
   - 创建中文字段含义字典

2. **探索性数据分析**：
   - 目标变量分布分析
   - 数值型变量可视化（直方图+箱线图）
   - 缺失值统计与可视化

3. **数据预处理**：
   - 缺失值处理（删除高缺失率字段+填充0）
   - 噪声处理（分箱平滑+移动平均平滑）
   - 数据变换（数值泛化+对数变换）

4. **特征工程**：
   - PCA降维（保留95%方差）
   - 特征选择（ANOVA F-value）

5. **结果导出**：
   - Excel格式保存原始数据和处理后数据
   - 字段字典持久化存储

## 环境要求

### Python依赖
```
numpy>=1.22.0
pandas>=1.4.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
openpyxl>=3.0.0
joblib>=1.1.0
missingno>=0.5.0  # 可选，用于缺失值可视化
```

安装所有依赖：
```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl joblib missingno
```

### 数据文件要求
项目根目录下需要`data`文件夹包含：
1. `data1.csv` - 行业类别数据
2. `data2.csv` - 财务数据
3. `data3.xlsx` - 字段含义说明

## 使用说明

### 运行项目
```python
python main.py
```

### 处理流程
1. **数据加载**：自动加载并合并数据源
2. **EDA分析**：生成可视化图表到`plots`目录
3. **数据预处理**：清洗、转换数据
4. **特征工程**：降维和特征选择
5. **结果导出**：保存处理结果到`output`目录

### 目录结构
```
项目根目录/
├── main.py                # 主程序
├── data/                  # 数据目录（必须）
│   ├── data1.csv          # 行业数据
│   ├── data2.csv          # 财务数据
│   └── data3.xlsx         # 字段说明
├── plots/                 # 自动生成-可视化图表
└── output/                # 自动生成-处理结果
```

## 输出文件

### plots目录（可视化图表）
| 文件名称 | 说明 |
|----------|------|
| `{字段名}_dist.png` | 单个字段的分布图（直方图+箱线图） |
| `numeric_distributions.png` | 所有数值字段分布图合集 |
| `missing_values_matrix.png` | 缺失值矩阵图（需安装missingno） |

### output目录（处理结果）
| 文件名称 | 内容 | 格式 |
|----------|------|------|
| `processed_data.xlsx` | 处理结果数据 | Excel |
| ┣ `原始数据` sheet | 清洗后的原始数据 |  |
| ┗ `处理后数据` sheet | 预处理后的数据 |  |
| `fields_dict.pkl` | 字段含义字典 | Pickle |

## 控制台输出示例

```
数据加载完成，形状: (5000, 120)
财务造假分布:
0    4500
1     500
Name: FLAG, dtype: int64

缺失值统计表:
            字段名  缺失值数量  缺失率(%)      数据类型
0  TICKER_SYMBOL        0     0.00       int64
1             AP      120     2.40     float64
...

删除了 15 个高缺失率字段
填充了 42 个字段的缺失值

主成分方差解释比例: [0.25, 0.18, 0.12,...]
累计方差解释比例: 0.95
PCA降维结果形状: (5000, 8)

筛选出的重要特征: ['AP', 'ADVANCE_RECEIPTS', 'TAXES_PAYABLE', ...]

结果已导出到 ./output 目录
处理流程完成
```

## 注意事项

1. 确保数据文件使用UTF-8编码，如有中文乱码可尝试`encoding='gbk'`
2. 首次运行会自动创建`plots`和`output`目录
3. 缺失值可视化需要安装`missingno`库
4. 项目默认筛选制造业数据，如需修改请调整`load_and_merge_data`方法

## 自定义配置

主要参数调整位置：
1. **行业筛选**：`load_and_merge_data`方法中的行业筛选条件
2. **可视化字段**：`exploratory_data_analysis`方法中的`numeric_cols`列表
3. **PCA保留方差**：`_pca_reduction`方法中的`n_components`参数
4. **特征选择数量**：`_filter_feature_selection`方法中的`k`参数



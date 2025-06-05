import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# 配置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['fangsong']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常问题

class FinancialDataPreprocessor:
    def __init__(self, data_paths):
        """
        初始化财务数据预处理器
        参数:
        data_paths (dict): 数据路径字典（键值对）:
            'category': 行业类别数据路径
            'financial': 财务数据路径
            'fields': 字段含义数据路径
        """
        self.data_paths = data_paths
        self.fields = {}  # 字段含义字典
        self.df = None  # 主数据集
        self.processed_df = None  # 处理后的数据
        
    def load_and_merge_data(self):
        """加载并合并数据"""
        # 1. 加载数据
        df_category = pd.read_csv(self.data_paths['category'], encoding='utf-8', on_bad_lines='skip')
        df_financial = pd.read_csv(self.data_paths['financial'], encoding='utf-8', on_bad_lines='skip')
        df_fields = pd.read_excel(self.data_paths['fields'])
        
        # 创建字段含义字典
        self.fields = dict(zip(df_fields['字段名'], zip(df_fields['含义'], df_fields['单位'])))
        
        # 2. 数据合并
        df_category.columns = ['TICKER_SYMBOL', '所属行业']
        self.df = pd.merge(df_financial, df_category, how='left', on=['TICKER_SYMBOL'])
        self.df.insert(1, 'INDUSTRY', self.df.pop('所属行业'))
        self.fields['INDUSTRY'] = ('所属行业', np.nan)
        
        # 3. 筛选制造业数据
        self.df = self.df[self.df['INDUSTRY'] == '制造业'].drop_duplicates().reset_index(drop=True)
        return self.df
    
    def exploratory_data_analysis(self, output_dir='plots'):
        """
        执行探索性数据分析(EDA)
        
        参数:
        output_dir 输出图表目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 打印基本信息
        print("数据形状:", self.df.shape)
        print("财务造假分布:\n", self.df['FLAG'].value_counts())
        
        # 选择部分数值型字段进行可视化
        numeric_cols = ['AP', 'ADVANCE_RECEIPTS', 'SOLD_FOR_REPUR_FA', 'PAYROLL_PAYABLE', 
                        'TAXES_PAYABLE', 'INT_PAYABLE', 'DIV_PAYABLE', 'OTH_PAYABLE']
        
        # 生成每个字段的直方图和箱线图
        for col in numeric_cols:
            self._plot_distribution(col, output_dir)
        
        # 数值型变量分布可视化
        self._plot_numeric_distributions(numeric_cols, output_dir)
        
        # 缺失值分析
        self._analyze_missing_values(output_dir)
        
    
    def preprocess_data(self):
        """执行数据预处理"""
        if self.df is None:
            raise ValueError("请先加载数据")
            
        df_processed = self.df.copy()
        
        # 缺失值处理
        df_processed = self._handle_missing_values(df_processed)
        
        # 噪声处理示例
        df_processed = self._bin_smooth(df_processed, 'BIZ_TAX_SURCHG', n_bins=5)
        df_processed = self._moving_average_smooth(df_processed, 'T_SH_EQUITY', window=3)
        
        # 数据泛化示例
        bins = [-float('inf'), 1e8, 1e9, float('inf')]
        labels = ['小型企业', '中型企业', '大型企业']
        df_processed = self._generalize_data(df_processed, 'N_INCOME', bins, labels)
        
        # 数值规约示例
        df_processed = self._reduce_numerical(df_processed, ['NOPERATE_INCOME'])
        
        self.processed_df = df_processed
        return df_processed
    
    def feature_engineering(self):
        """执行特征工程"""
        if self.processed_df is None:
            raise ValueError("请先预处理数据")
            
        # 数据降维
        numeric_cols = [col for col in self.processed_df.columns 
                        if pd.api.types.is_numeric_dtype(self.processed_df[col]) 
                        and col != 'FLAG']
        pca_result = self._pca_reduction(self.processed_df, numeric_cols)
        
        # 特征选择
        drop_cols = ['TICKER_SYMBOL', 'INDUSTRY', 'ACCOUTING_STANDARDS', 'N_INCOME_generalized', 
                    'REPORT_TYPE', 'CURRENCY_CD', 'BIZ_TAX_SURCHG_bin', 'BIZ_TAX_SURCHG_smooth', 
                    'T_SH_EQUITY_smooth', 'log_NOPERATE_INCOME']
        
        feature_cols = [col for col in self.processed_df.columns 
                        if col not in (['FLAG'] + drop_cols) 
                        and pd.api.types.is_numeric_dtype(self.processed_df[col])]
        
        if 'FLAG' in self.processed_df.columns and self.processed_df['FLAG'].notnull().sum() > 0:
            X = self.processed_df[feature_cols]
            y = self.processed_df['FLAG']
            selected_features = self._filter_feature_selection(X, y)
            return pca_result, selected_features
        
        return pca_result, None
    
    def export_results(self, output_dir='output'):
        """导出处理结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        with pd.ExcelWriter(f'{output_dir}/processed_data.xlsx') as writer:
            if self.df is not None:
                self.df.to_excel(writer, sheet_name='原始数据', index=False)
            
            if self.processed_df is not None:
                self.processed_df.to_excel(writer, sheet_name='处理后数据', index=False)
        
        # 保存字段含义字典
        joblib.dump(self.fields, f'{output_dir}/fields_dict.pkl')
        
        print(f"结果已导出到 {output_dir} 目录")
    
    #辅助工具函数
    def _plot_distribution(self, col, output_dir):
        """绘制单个字段的分布图"""
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(self.df[col], kde=True)
        plt.title(f'{self.fields[col][0]}直方图')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=self.df[col])
        plt.title(f'{self.fields[col][0]}箱线图')
        
        plt.savefig(f'{output_dir}/{col}_dist.png')
        plt.close()
    
    def _plot_numeric_distributions(self, cols, output_dir, n_cols=4):
        """绘制多个数值变量的分布"""
        n_rows = (len(cols) + n_cols - 1) // n_cols
        plt.figure(figsize=(20, 5 * n_rows))
        
        for i, col in enumerate(cols):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.histplot(self.df[col], kde=True, bins=30)
            plt.title(f'{self.fields[col][0]}分布', fontsize=12)
            plt.xlabel('')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/numeric_distributions.png')
        plt.close()
    
    def _analyze_missing_values(self, output_dir):
        """分析并可视化缺失值"""
        # 缺失值统计
        data_null = self._miss_data_count(self.df)
        print("缺失值统计表:\n", data_null)
        
        # 缺失值可视化
        try:
            import missingno as msno
            msno.matrix(self.df, figsize=(15, 10))
            plt.savefig(f'{output_dir}/missing_values_matrix.png')
            plt.close()
        except ImportError:
            print("库异常")
    
    def _miss_data_count(self, data):
        """生成缺失值统计表"""
        null_count = data.isnull().sum().to_frame('缺失值数量')
        null_count['缺失率(%)'] = (null_count['缺失值数量'] / len(data) * 100).round(2)
        null_count['数据类型'] = data.dtypes
        return null_count.reset_index().rename(columns={'index': '字段名'})
    
    def _handle_missing_values(self, df):
        """处理缺失值"""
        data_null = self._miss_data_count(df)
        
        # 确定处理策略
        drop_cols = data_null[data_null['缺失率(%)'] >= 70]['字段名'].tolist()
        fill0_cols = data_null[(data_null['缺失率(%)'] > 0) & (data_null['缺失率(%)'] < 70)]['字段名'].tolist()
        
        # 执行处理
        df = df.drop(columns=drop_cols)
        df[fill0_cols] = df[fill0_cols].fillna(0)
        
        print(f"删除了 {len(drop_cols)} 个高缺失率字段")
        print(f"填充了 {len(fill0_cols)} 个字段的缺失值")
        return df
    
    def _bin_smooth(self, df, col, n_bins=5, method='mean'):
        """分箱平滑处理"""
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notnull().all():
            try:
                df[f'{col}_bin'] = pd.cut(df[col], bins=n_bins)
                
                if method == 'mean':
                    smooth_values = df.groupby(f'{col}_bin')[col].transform('mean')
                elif method == 'median':
                    smooth_values = df.groupby(f'{col}_bin')[col].transform('median')
                else:
                    return df
                
                df[f'{col}_smooth'] = smooth_values
            except Exception as e:
                print(f"分箱平滑处理出错: {e}")
        return df
    
    def _moving_average_smooth(self, df, col, window=3):
        """移动平均光滑处理"""
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f'{col}_smooth'] = df[col].rolling(window=window, min_periods=1).mean()
        return df
    
    def _generalize_data(self, df, col, bins, labels):
        """数据泛化处理"""
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f'{col}_generalized'] = pd.cut(df[col], bins=bins, labels=labels, right=False)
        return df
    
    def _reduce_numerical(self, df, cols):
        """数值规约处理"""
        for col in cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # 处理负值问题
                if (df[col] < 0).any():
                    df[f'log_{col}'] = np.log1p(df[col].apply(lambda x: x if x >= 0 else 0))
                else:
                    df[f'log_{col}'] = np.log1p(df[col])
        return df
    
    def _pca_reduction(self, df, cols, n_components=0.95):
        """PCA降维"""
        numeric_cols = [col for col in cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_cols:
            print("没有找到适合PCA的数值型字段")
            return None
        
        X = df[numeric_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
        
        print(f'主成分方差解释比例: {pca.explained_variance_ratio_}')
        print(f'累计方差解释比例: {sum(pca.explained_variance_ratio_)}')
        
        return df_pca
    
    def _filter_feature_selection(self, X, y, k=10):
        """过滤式特征选择"""
        valid_indices = y.notnull()
        X_clean = X[valid_indices].fillna(0)
        y_clean = y[valid_indices]
        
        if X_clean.empty or y_clean.empty:
            print("清洗后数据不足，无法进行特征选择")
            return []
        
        selector = SelectKBest(score_func=f_classif, k=min(k, X_clean.shape[1]))
        try:
            selector.fit(X_clean, y_clean)
            selected_features = X.columns[selector.get_support()]
            print(f'筛选出的重要特征: {list(selected_features)}')
            return selected_features
        except Exception as e:
            print(f"特征选择出错: {e}")
            return []
    
# 主函数
def main():
    # 配置数据路径
    data_paths = {
        'category': 'data/data1.csv',
        'financial': 'data/data2.csv',
        'fields': 'data/data3.xlsx'
    }
    
    # 创建预处理器实例
    preprocessor = FinancialDataPreprocessor(data_paths)
    
    # 1. 加载并合并数据
    raw_data = preprocessor.load_and_merge_data()
    print("数据加载完成，形状:", raw_data.shape)
    
    # 2. 探索性数据分析
    preprocessor.exploratory_data_analysis('./plots')
    
    # 3. 数据预处理
    processed_data = preprocessor.preprocess_data()
    print("数据预处理完成，形状:", processed_data.shape)
    
    # 4. 特征工程
    pca_result, selected_features = preprocessor.feature_engineering()
    if pca_result is not None:
        print("PCA降维结果形状:", pca_result.shape)
    if selected_features is not None:
        print("筛选的特征:", selected_features)
    
    # 5. 导出结果
    preprocessor.export_results('./output')
    print("处理流程完成")

if __name__ == "__main__":
    main()

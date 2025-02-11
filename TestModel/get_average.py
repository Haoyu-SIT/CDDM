import pandas as pd

# 假设'file_path'是你的文件路径
file_path = '/scratch/iu60/hs2870/results/20240229-test/2009/BrownianBridge/samples/0/test_sample'

def calculate_metrics_average(file_path):
    # 读取文件
    df = pd.read_csv(file_path)
    print(df.columns)
    df.columns = df.columns.str.strip()
    print(df.columns)
    # 删除包含重复列名的行
    # 检查所有目标列是否包含它们自己的列名作为数据，如果是，则删除这些行
    columns_to_clean = ['MSE', 'RMSE', 'MAE', '95 Percentile', '99 Percentile']
    for column in columns_to_clean:
        df[column] = df[column].str.replace('[^\d.]', '', regex=True)
        df[column] = pd.to_numeric(df[column], errors='coerce')
    for column in df.columns:
        df = df[df[column] != column]
    
    # 确保列名没有多余的空格
    df.columns = df.columns.str.strip()
    
    # 计算各项指标的平均值
    mse_mean = df['MSE'].mean()
    rmse_mean = df['RMSE'].mean()
    mae_mean = df['MAE'].mean()
    percentile_95_mean = df['95 Percentile'].mean()
    percentile_99_mean = df['99 Percentile'].mean()
    
    # 打印平均值
    print(f"MSE平均值: {mse_mean}")
    print(f"RMSE平均值: {rmse_mean}")
    print(f"MAE平均值: {mae_mean}")
    print(f"95百分位数平均值: {percentile_95_mean}")
    print(f"99百分位数平均值: {percentile_99_mean}")

# 替换为你的实际文件路径
calculate_metrics_average(file_path)

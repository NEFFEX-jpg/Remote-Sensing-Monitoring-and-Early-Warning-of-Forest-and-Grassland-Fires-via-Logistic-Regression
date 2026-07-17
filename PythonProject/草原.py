import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  classification_report,roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import  ttest_ind
import rasterio as rio
import numpy as np

def main():
#一，森林火险模型训练
    # 1. 加载数据
    file_path = "C:/Users/17873/Desktop/traindata.txt"
    data = pd.read_csv(file_path)
    # 2. 数据预处理
    data=data.dropna(axis=0)#删除缺失值（假设缺失值较少）
    print("数据加载完成，数据形状：", data.shape)
    #print(data.isnull().sum())
    # 独热编码处理离散变量
    aspect_encoded = pd.get_dummies(data['aspect'],dtype=int, prefix='aspect')
    data = pd.concat([data, aspect_encoded], axis=1)
    data = data.drop('aspect', axis=1)
    # 特征选择（假设目标列为 'tag'，其余列为特征）
    features = [   'LFMC8', '16_8dif','LAI', 'altitude', 'slope',  'wind','temperature', 'rain',
                    'humid']+list(aspect_encoded.columns)
    target = 'tag'

    data=data[data['ZGBP'].isin([1,2,3,4,5,6,7,8])]#森林类型
    X = data[features]
    y = data[target].values.ravel()
    print("特征形状：", X.shape)
    print("目标形状：", y.shape)
    #相关性分析
    correlation_matrix = X.corr(method='pearson')
    plt.rcParams['font.sans-serif']=['SimHei']#设置字体为黑体
    plt.rcParams['axes.unicode_minus']=False#解决负号显示问题
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True,annot_kws={"size":8}, cmap='coolwarm', center=0)
    plt.title('森林模型特征之间的皮尔逊相关性热力图')
    plt.show()
    #显著性差异分析
    fire_data = data[data[target] == 1]
    non_fire_data = data[data[target] == 0]
    significance_results = {}
    for feature in features:
        t_stat, p_value = ttest_ind(fire_data[feature], non_fire_data[feature], equal_var=False)
        significance_results[feature] = (t_stat, p_value)
    print("\n森林模型显著性差异分析结果：")
    for feature, (t_stat, p_value) in significance_results.items():
        print(f"特征：{feature}")
        print(f"t 统计量：{t_stat:.4f}")
        print(f"p 值：{p_value:.4f}")
        print("显著性结论：", "显著" if p_value < 0.05 else "不显著")
        print("-------------------------")


    # 3. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print("训练集形状：", X_train.shape)
    print("测试集形状：", X_test.shape)

    # 4. 特征标准化
    scaler1 = StandardScaler()
    X_train_scaled = scaler1.fit_transform(X_train)
    X_test_scaled = scaler1.transform(X_test)

    # 5. 训练逻辑回归模型
    model1 = LogisticRegression(max_iter=800,class_weight='balanced',penalty='l2')
    model1.fit(X_train_scaled, y_train)
    print("模型训练完成！")
    coefficients=model1.coef_
    intercept=model1.intercept_
    print("森林模型各特征系数：\n",coefficients)
    print("森林模型截距项",intercept)

    # 6. 模型评估
    y_pred = model1.predict(X_test_scaled)
    print("森林模型分类报告：")
    print(classification_report(y_test, y_pred))

    y_proba = model1.predict_proba(X_test_scaled)[:, 1]  # 获取正类的概率
    auc_score = roc_auc_score(y_test, y_proba)
    print("\nAUC 分数：", auc_score)
    #/绘制曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
#二，草原火险模型训练
    # 1. 加载数据
    file_path = "C:/Users/17873/Desktop/traindata.txt"
    data = pd.read_csv(file_path)
    # 2. 数据预处理
    data = data.dropna(axis=0)  # 删除缺失值（假设缺失值较少）
    print("数据加载完成，数据形状：", data.shape)
    # print(data.isnull().sum())

    # 独热编码处理离散变量
    aspect_encoded = pd.get_dummies(data['aspect'], dtype=int, prefix='aspect')
    data = pd.concat([data, aspect_encoded], axis=1)
    data = data.drop('aspect', axis=1)
    # 特征选择（假设目标列为 'tag'，其余列为特征）
    features = ['LFMC8', '16_8dif', 'LAI', 'altitude', 'slope', 'wind', 'temperature',
                'humid',] + list(aspect_encoded.columns)
    target = 'tag'

    data = data[data['ZGBP'].isin([9, 10, 11])]  # 草原类型
    X = data[features]
    y = data[target].values.ravel()

    print("特征形状：", X.shape)
    print("目标形状：", y.shape)
    # 相关性分析
    correlation_matrix = X.corr(method='pearson')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, annot_kws={"size": 8}, cmap='coolwarm', center=0)
    plt.title('草原模型特征之间的皮尔逊相关性热力图')
    plt.show()
    # 显著性差异分析
    fire_data = data[data[target] == 1]
    non_fire_data = data[data[target] == 0]
    significance_results = {}
    for feature in features:
        t_stat, p_value = ttest_ind(fire_data[feature], non_fire_data[feature], equal_var=False)
        significance_results[feature] = (t_stat, p_value)
    print("\n草原模型显著性差异分析结果：")
    for feature, (t_stat, p_value) in significance_results.items():
        print(f"特征：{feature}")
        print(f"t 统计量：{t_stat:.4f}")
        print(f"p 值：{p_value:.4f}")
        print("显著性结论：", "显著" if p_value < 0.05 else "不显著")
        print("-------------------------")

    # 3. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print("训练集形状：", X_train.shape)
    print("测试集形状：", X_test.shape)

    # 4. 特征标准化
    scaler2 = StandardScaler()
    X_train_scaled = scaler2.fit_transform(X_train)
    X_test_scaled = scaler2.transform(X_test)

    # 5. 训练逻辑回归模型
    model2 = LogisticRegression(max_iter=800, class_weight='balanced',penalty='l2')
    model2.fit(X_train_scaled, y_train)
    print("草原模型训练完成！")
    coefficients = model2.coef_
    intercept = model2.intercept_
    print("草原模型各特征系数：\n", coefficients)
    print("草原模型截距项", intercept)

    # 6. 模型评估
    y_pred = model2.predict(X_test_scaled)
    print("草原模型分类报告：")
    print(classification_report(y_test, y_pred))

    y_proba = model2.predict_proba(X_test_scaled)[:, 1]  # 获取正类的概率
    auc_score = roc_auc_score(y_test, y_proba)
    print("\nAUC 分数：", auc_score)
    # /绘制曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

#三，读取数据，预测火险，绘图
    #1，存储所有图片地址到列表中
    tif_files = ["C:/Users/17873/Desktop/test/2018001_FMCchongcaiyang.tif",  # LFMC16
                 "C:/Users/17873/Desktop/test/2018009_FMCchongcaiyang.tif",  # LFMC8
                 "C:/Users/17873/Desktop/test/aspect.tif",  # aspect0
                 "C:/Users/17873/Desktop/test/elevation.tif",  # altitude
                 "C:/Users/17873/Desktop/test/humid.tif",  # humid
                 "C:/Users/17873/Desktop/test/MCD12Q1.A2018001.tif",  # ZGBP
                 "C:/Users/17873/Desktop/test/MCD15A2H.A2018009chongcaiyang.tif",  # LAI
                 "C:/Users/17873/Desktop/test/slope.tif",  # slope
                 "C:/Users/17873/Desktop/test/temp.tif",  # temperature
                 "C:/Users/17873/Desktop/test/wind.tif",  # wind
                 "C:/Users/17873/Desktop/test/rain.tif"  # rain
                 ]
    #2，循环读取地址，将信息存储到列表中
    tif_data = []
    image_path="C:/Users/17873/Desktop/test/2018001_FMCchongcaiyang.tif"
    profile=rio.open(image_path).profile
    for file_path in tif_files:
        with rio.open(file_path) as geoimg:
            img_arr = geoimg.read()
            img_arr_1d = img_arr.reshape(-1)#一维化
            tif_data.append(img_arr_1d)
    df = pd.DataFrame(tif_data).T#将列表转换为DataFrame格式，重命名，
    df.columns = ['LFMC16', 'LFMC8', 'aspect0', 'altitude', 'humid', 'ZGBP', 'LAI', 'slope', 'temperature', 'wind',
                  'rain']
    df['16_8dif'] = df['LFMC16'] - df['LFMC8']
    df['LAI'] = df['LAI'] * 0.1
    # 处理异常值，LAI值一般小于10，大于10的部分置为0
    df.loc[df['LAI'] > 10, 'LAI'] = 0
    df.loc[df['aspect0'] < 0, 'aspect0'] = 0
    # 3，坡向离散化并独热编码
    # 定义区间
    bins = [-0.1, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
    # 定义标签
    labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0]
    df['aspect'] = pd.cut(df['aspect0'], bins=bins, labels=labels, ordered=False)
    # 独热编码处理离散变量
    aspect_encoded = pd.get_dummies(df['aspect'], dtype=int, prefix='aspect')
    df = pd.concat([df, aspect_encoded], axis=1)
    #4，处理森林数据
    features = ['LFMC8', '16_8dif','LAI', 'altitude', 'slope',  'wind','temperature', 'rain',
                'humid']+list(aspect_encoded.columns)
    X=df[features]
    X_scaled=scaler1.transform(X)
    y=model1.predict_proba(X_scaled)[:,1]
    df['prediction1']=y
    df.loc[~df['ZGBP'].isin([1,2,3,4,5,6,7,8]), 'prediction1'] = 0
    #5，处理草原数据
    features = ['LFMC8', '16_8dif', 'LAI', 'altitude', 'slope', 'wind', 'temperature',
                'humid'] + list(aspect_encoded.columns)
    X = df[features]
    X_scaled = scaler2.transform(X)
    y = model2.predict_proba(X_scaled)[:,1]
    df['prediction2'] = y
    #6，将森林草原概率结合在一列
    df.loc[~df['ZGBP'].isin([9,10,11]), 'prediction2'] = 0
    df['prediction']=df['prediction1']+df['prediction2']
    df.loc[df['prediction'] < 0, 'prediction'] = 0#设置最小值为0
    df.loc[~df['ZGBP'].isin([1,2,3,4,5,6,7,8,9,10,11]), 'prediction'] = -0.005#设置无效值
    profile.update({'nodata': 0})
    #转换为二维列表
    a = df['prediction'].values
    pred_img = a.reshape((profile['height'], profile['width']))
    # 7. 制作火险概率图
    # 获取原始红黄绿颜色映射
    original_cmap = plt.get_cmap('RdYlGn_r')
    # 创建新颜色列表：起始为白色，后续保留rdylbu渐变
    new_colors = original_cmap(np.linspace(0, 1, 256))
    new_colors[0, :] = [1, 1, 1, 1]  # 第一个颜色设为白色
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_RdYlGn_r', new_colors)
    plt.figure(figsize=(12, 8))
    plt.imshow(pred_img, cmap=custom_cmap, vmin=-0.005, vmax=1)
    plt.colorbar(label='Fire Probability')
    plt.title('火险概率地图')
    plt.axis('off')
    plt.show()
if __name__ == "__main__":
    main()
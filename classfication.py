import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

# 生成合成分类数据集
X, y = make_classification(n_samples=200,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           weights=[0.5, 0.5],
                           random_state=17)

# 训练多层感知器（MLP）神经网络分类器
model = MLPClassifier(hidden_layer_sizes=(12,),
                      activation='relu',
                      solver='adam',
                      max_iter=5000,
                      random_state=17)
model.fit(X, y)

# 绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)

# 获取坐标轴范围
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# 生成网格点坐标矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测分类结果并绘制决策边界
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

# 设置图形标题和坐标轴标签
plt.title('Nonlinear Decision Boundary (MLP)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 显示图形
plt.show()

outliers = np.array([
    [-2, 0.4],  # 另类数据点1
    [-2.2, 0.4],  # 另类数据点2
    [-2.4, 0.4],  # 另类数据点3
    [-2.6, 0.4],  # 另类数据点4
    [-2.8, 0.4]  # 另类数据点5
])
label = np.array([0, 0, 0, 0, 0])
px = np.concatenate((X, outliers), axis=0)
py = np.append(y, label)
model.fit(px, py)
# 绘制散点图
plt.scatter(px[:, 0], px[:, 1], c=py, cmap='bwr', alpha=0.7)

# 获取坐标轴范围
x_min, x_max = px[:, 0].min() - 1, px[:, 0].max() + 1
y_min, y_max = px[:, 1].min() - 1, px[:, 1].max() + 1

# 生成网格点坐标矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测分类结果并绘制决策边界
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

# 设置图形标题和坐标轴标签
plt.title('Nonlinear Decision Boundary (MLP)p')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 显示图形
plt.show()

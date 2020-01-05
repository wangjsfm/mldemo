import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split



originData = pd.read_excel('./resource/predict.xlsx')
x = originData.loc[:,['mw','so2']]
y = originData.loc[:,['target']]
# y=[]
# for item in y_temp:
#     y.append(item[0])
#     print(item[0])


x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=0.02)




# 回归部分
def try_different_method(model, method):
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    plt.figure()
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(np.arange(len(result)), y_test, "go-", label="True value")
    plt.plot(np.arange(len(result)), result, "ro-", label="Predict value")
    plt.title(f"method:{method}---score:{score}")
    plt.legend(loc="best")
    plt.show()
    return  score

ModelList=[]
# 方法选择
# 1.决策树回归
from sklearn import tree

model_decision_tree_regression = tree.DecisionTreeRegressor()
ModelList.append([model_decision_tree_regression,'决策树回归'])

# 2.线性回归
from sklearn.linear_model import LinearRegression

model_linear_regression = LinearRegression()
ModelList.append([model_linear_regression,'线性回归'])

# 3.SVM回归
from sklearn import svm

model_svm = svm.SVR()
ModelList.append([model_svm,'SVM回归'])


# 4.kNN回归
from sklearn import neighbors

model_k_neighbor = neighbors.KNeighborsRegressor()
ModelList.append([model_k_neighbor,'kNN回归'])


# 5.随机森林回归
from sklearn import ensemble

model_random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=20)  # 使用20个决策树
ModelList.append([model_random_forest_regressor,'随机森林回归'])


# 6.Adaboost回归
from sklearn import ensemble

model_adaboost_regressor = ensemble.AdaBoostRegressor(n_estimators=50)  # 这里使用50个决策树
ModelList.append([model_adaboost_regressor,'Adaboost回归'])


# 7.GBRT回归
from sklearn import ensemble

model_gradient_boosting_regressor = ensemble.GradientBoostingRegressor(n_estimators=100)  # 这里使用100个决策树
ModelList.append([model_gradient_boosting_regressor,'GBRT回归'])


# 8.Bagging回归
from sklearn import ensemble

model_bagging_regressor = ensemble.BaggingRegressor()
ModelList.append([model_bagging_regressor,'Bagging回归'])


# 9.ExtraTree极端随机数回归
from sklearn.tree import ExtraTreeRegressor

model_extra_tree_regressor = ExtraTreeRegressor()
ModelList.append([model_extra_tree_regressor,'ExtraTree极端随机数回归'])


def modelSelect():
    scores = []
    for item in ModelList:
        scores.append([try_different_method(item[0], item[1]), item[1]])
    return  scores

if __name__ == '__main__':
    scores = []
    for item in range(1): #训练次数，做对比
        scores.append(modelSelect())

    for item in scores: #打印得分
        print(item)


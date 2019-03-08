# coding: utf-8
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.ensemble import BaggingRegressor
import warnings
warnings.filterwarnings("ignore")
from matplotlib.pylab import rcParams
import matplotlib.pylab as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score,train_test_split
data_train = pd.read_csv('train.csv', engine='python', encoding='UTF-8')
from sklearn.ensemble import RandomForestRegressor
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:, 0]
    X= known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X,y)
    predictedAges = rfr.predict(unknown_age[:, 1:])
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    return df,rfr
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
    return df
data_train,rfr = set_missing_ages(data_train)
df = set_Cabin_type(data_train)
#dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
#dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
#dummies_Sex = pd.get_dummies(data_train['Sex'],prefix='Sex')
#dummies_Pclass = pd.get_dummies(data_train['Pclass'],prefix='Pclass')
#df = pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)

df.loc[(df.Sex =='male'),'Sex'] = 1
df.loc[(df.Sex=='female'),'Sex'] = 2
df.loc[(df.Cabin =='Yes'),'Cabin'] = 1
df.loc[(df.Cabin =='No'),'Cabin'] = 2
df.loc[(df.Embarked =='S'),'Embarked'] = 2
df.loc[(df.Embarked =='C'),'Embarked'] = 1
df.loc[(df.Embarked =='Q'),'Embarked'] = 2
df.loc[(df.Embarked.isnull()),'Embarked'] =2
df['MyAge'] = df.Age.apply(lambda x:1 if x<=12 else 0) #根据某一列生成一列
def function1(a,b):
    if a == 1 and b==0:
        return 1
    else:return 2
df['Lady'] = df.apply(lambda x: function1(x.Pclass,x.Sex),axis=1) #根据多列生成一列
def function2(a,b):
    if 'Mrs' in a and b>=1:
        return 1
    else:return 2
df['Mother'] = df.apply(lambda x: function2(x.Name,x.Parch),axis=1) #根据多列生成一列
df['FamilySize'] = df['SibSp'] + df['Parch'] +1
df.loc[(df.FamilySize>=4),'Sizetoobig'] = 1
df.loc[(df.FamilySize<4),'Sizetoobig'] = 2
def function3(a):
    if 'Mrs' in a or 'Miss' in a:
        return 1
    else:return 2
df['Woman'] = df.apply(lambda  x: function3(x.Name),axis=1)

df.loc[ df['Age'] <= 16, 'Age'] = 0
df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
df.loc[ df['Age'] > 64, 'Age'] = 4
df['Age'] = df['Age'].astype(int)



from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
train_df = df.filter(regex='Survived|Pclass|Sex|Age|Fare|Lady|Embarked|Sizetoobig|Woman|FamilySize|Cabin')

train_np = train_df.as_matrix()

y = train_np[:, 0]
X = train_np[:, 1:]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

clf = XGBClassifier(learning_rate=0.15,
                     n_estimators=120
                     ,max_depth=6,
                     min_child_weight=5,
                     gamma=0.35,
                     subsample=0.9,
                     colsample_bytree=0.8,
                     objective='binary:logistic',
                     nthread=4,
                     scale_pos_weight=1,
                     seed=27)

#bagging_clf = BaggingRegressor(clf,n_estimators=18,max_samples=0.8,max_features=1.0,bootstrap=True,bootstrap_features=False,n_jobs=-1)
#param_tests1 = {'learning_rate':[i/20.0 for i in range(0,10)]}
#gsearch = GridSearchCV(estimator=clf,param_grid=param_tests1,n_jobs=4,cv=5)
#gsearch.fit(X,y)
#print(gsearch.best_params_)

clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print(accuracy_score(y_train,clf.predict(X_train)))
print(accuracy_score(y_test,predictions))
print(np.mean(cross_val_score(clf,X,y,cv=5)))
"""
split_train,split_cv = train_test_split(df,test_size=0.2,random_state=1)
origin_data_train = pd.read_csv("train.csv")
bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].
    isin(split_cv[predictions != y_test]['PassengerId'].values)]
#print(bad_cases)#找出bad cases
"""
"""
def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(.05,1.,20),verbose=0,plot=True):
    train_sizes,train_scores,test_scores = learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes,verbose=verbose)
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff
plot_learning_curve(clf,"学习曲线",X,y)

"""

"""
####对测试集进行同样的处理
data_test = pd.read_csv('test.csv')
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)



#data_test.drop(['Name','Ticket'],axis=1,inplace=True)
data_test.loc[(data_test.Sex =='male'),'Sex'] = 1
data_test.loc[(data_test.Sex=='female'),'Sex'] = 2
data_test.loc[(data_test.Cabin =='Yes'),'Cabin'] = 1
data_test.loc[(data_test.Cabin =='No'),'Cabin'] = 2
data_test.loc[(data_test.Embarked =='S'),'Embarked'] = 2
data_test.loc[(data_test.Embarked =='C'),'Embarked'] = 1
data_test.loc[(data_test.Embarked =='Q'),'Embarked'] = 2
data_test.loc[(data_test.Embarked.isnull()),'Embarked'] = 2
data_test['MyAge'] = data_test.Age.apply(lambda x:1 if x<=12 else 0)
data_test['Lady'] = data_test.apply(lambda x: function1(x.Pclass,x.Sex),axis=1)
data_test['Mother'] = data_test.apply(lambda x: function2(x.Name,x.Parch),axis=1) #根据多列生成一列

data_test['FamilySize'] = data_test['SibSp'] + data_test['Parch'] +1
data_test.loc[(data_test.FamilySize>=4),'Sizetoobig'] = 1
data_test.loc[(data_test.FamilySize<4),'Sizetoobig'] = 2
def function3(a):
    if 'Mrs' in a or 'Miss' in a:
        return 1
    else:return 2
data_test['Woman'] = data_test.apply(lambda  x: function3(x.Name),axis=1)

data_test.loc[ data_test['Age'] <= 16, 'Age'] = 0
data_test.loc[(data_test['Age'] > 16) & (data_test['Age'] <= 32), 'Age'] = 1
data_test.loc[(data_test['Age'] > 32) & (data_test['Age'] <= 48), 'Age'] = 2
data_test.loc[(data_test['Age'] > 48) & (data_test['Age'] <= 64), 'Age'] = 3
data_test.loc[ data_test['Age'] > 64, 'Age'] = 4
data_test['Age'] = data_test['Age'].astype(int)




test = data_test.filter(regex='Survived|Pclass|Sex|Age|Fare|Lady|Embarked|Sizetoobig|Woman|FamilySize|Cabin')
predictions = clf.predict(test.as_matrix())
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("result.csv", index=False)
#print(pd.DataFrame({'columns':list(train_df.columns)[1:], 'coef':list(clf.coef_.T)}))
"""
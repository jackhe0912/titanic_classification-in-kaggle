import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score



def get_titanic_data():
    trainData=pd.read_csv('train.csv')
    testData=pd.read_csv('test.csv')

    return trainData,testData

def drop_feature(dataset):
    """
    :param traindata: 输入数据集
    :param feature:  输入特征
    :return: 返回丢弃特征后的数据
    """
    dataset.drop(['Ticket'],axis=1,inplace=True)

    return dataset

def Ebarked_transform(dataset):
    """
    处理Enbarked的缺失值
    :param dataset:
    :return: dataset

    """
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked']=dataset['Embarked']
    return dataset

def Cabin_transform(dataset):
    """
    对Cabin进行缺失值填充，并进行数值转换
    :param dataset:
    :return: dataset
    """
    dataset.loc[dataset['Cabin'].notnull(), 'Cabin'] = 1
    dataset.loc[dataset['Cabin'].isnull(), 'Cabin'] = 0

    return dataset
def age_trasform(dataset):
    """
    先对Age进行缺失值处理，然后进行离散化
    :param dataset:
    :return: dataset
    """
    mean_age=dataset['Age'].mean()  #求均值
    std_age=dataset['Age'].std()  # 求标准差
    nan_age_count=dataset['Age'].isnull().sum()   # 求出空值的数量

    rand = np.random.randint(mean_age - std_age, mean_age + std_age, size=nan_age_count)
    dataset.loc[dataset['Age'].isnull(), 'Age'] = rand
    dataset['Age'] = dataset['Age'].astype(int)

    dataset['Age'][(dataset['Age'] <= 16)] = 0
    dataset['Age'][(dataset['Age'] <= 32) & (dataset['Age'] > 16)] = 1
    dataset['Age'][(dataset['Age'] <= 48) & (dataset['Age'] > 32)] = 2
    dataset['Age'][(dataset['Age'] <= 64) & (dataset['Age'] > 48)] = 3
    dataset['Age'][(dataset['Age'] > 64)] = 4

    return dataset

def Fare_transform(dataset):
    """
    对Fare的值进行缺失值处理，并进行离散化
    :param dataset:
    :return:
    """
    dataset['Fare'].fillna(dataset.Fare.median(), inplace=True)
    dataset['Fare'][(dataset['Fare'] <= 7.75)] = 0
    dataset['Fare'][(dataset['Fare'] <= 14.75) & (dataset['Fare'] > 7.75)] = 1
    dataset['Fare'][(dataset['Fare'] <= 31) & (dataset['Fare'] > 14.75)] = 2
    dataset['Fare'][(dataset['Fare'] > 31)] = 3

    return dataset
def new_Family(dataset):
    """
    将SibSp和Parch组合在一起创建新的Family特征
    :param dataset:
    :return:
    """
    dataset['Family'] = dataset['SibSp'] + dataset['Parch']
    dataset['Family'].loc[dataset['Family'] == 0] = 0
    dataset['Family'].loc[dataset['Family'] > 0] = 1

    dataset = dataset.drop(['SibSp', 'Parch'], axis=1)

    return dataset

def new_Person(dataset):
    """
    根据 年龄和性别分为child,female,male 创建Person特征，
    :param dataset:
    :return:
    """
    dataset['Person']=np.nan
    dataset['Person'][dataset['Age']<16]='child'
    dataset['Person'][(dataset['Age']>16) &(dataset['Sex']=='female')]='female'
    dataset['Person'][(dataset['Age']>16)& (dataset['Sex']=='male')]='male'

    dataset.drop('Sex', axis=1, inplace=True)

    return dataset

def get_title(x):
    list = ['Mr.', 'Mrs.', 'Miss.', 'Master.']
    if x in list:
        return x
    else:
        return 'other'
def new_title(dataset):
    """
    根据称呼建立特征
    :param dataset:
    :return:
    """
    dataset['title'] = dataset['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    dataset['title']=dataset['title'].apply(get_title)
    dataset.drop('Name',axis=1,inplace=True)

    return dataset

def get_dummies(dataset):
    """
    对所有特征进行虚拟编码get_dummies
    :param dataset:
    :return:
    """
    title_dummies=pd.get_dummies(dataset['title'], prefix='title')
    Person_dummies=pd.get_dummies(dataset['Person'],prefix='Person')
    Family_dummies=pd.get_dummies(dataset['Family'],prefix='Family')
    Fare_dummies=pd.get_dummies(dataset['Fare'],prefix='Fare')
    Age_dummies=pd.get_dummies(dataset['Age'],prefix='Age')
    Cabin_dummies=pd.get_dummies(dataset['Cabin'],prefix='Cabin')
    Embarked_dummies=pd.get_dummies(dataset['Embarked'],prefix='Embarked')
    Pclass_dummies=pd.get_dummies(dataset['Pclass'],prefix='Pclass')

    dataset=pd.concat([dataset,title_dummies,Person_dummies,Family_dummies,Fare_dummies,Age_dummies,Cabin_dummies,Embarked_dummies,
                       Pclass_dummies],axis=1)
    dataset=dataset.drop(['title','Person','Family','Fare','Age','Cabin','Embarked','Pclass'],axis=1)

    return dataset

def collect(dataset):
    """
    将所有数据处理集合到一起
    :param dataset:
    :return:
    """
    dataset=drop_feature(dataset)
    dataset=Ebarked_transform(dataset)
    dataset=Cabin_transform(dataset)
    dataset=age_trasform(dataset)
    dataset=Fare_transform(dataset)
    dataset=new_Family(dataset)
    dataset=new_Person(dataset)
    dataset=new_title(dataset)
    dataset=get_dummies(dataset)

    return dataset

def train(dataset,clf):
    """
    训练模型
    :param dataset:数据集
    :param clf:分类器
    :return:
    """
    X_train_data=dataset .drop(['Passenger','Survived'], axis=1)
    Y_train_data=dataset['Survived']

    clf.fit(X_train_data,Y_train_data)
    score=clf.score(X_train_data,Y_train_data)
    cv_score = cross_val_score(clf, X_train_data, Y_train_data, cv=5)  #交叉验证

    return score,cv_score,clf

def test(dataset,clf):
    """
    测试集
    :param dataset:
    :param clf:
    :return:
    """
    X_test = dataset.drop('Passenger',axis=1)
    predictions = clf.predict(X_test)
    submission = pd.DataFrame({"PassengerId": dataset["PassengerId"],"Survived": predictions})
    return submission.to_csv('titanic.csv', index=False)

def main():

    data_train,data_test=get_titanic_data()
    data_train=collect(data_train)
    data_test=collect(data_test)
    models = {
        'LogisticReg': LogisticRegression(max_iter=500, tol=0.0001, penalty='l2', solver='lbfgs'),
        'svc': SVC(max_iter=200, kernel='rbf', gamma=0.5, C=5),
        'KNN': KNeighborsClassifier(n_neighbors=9),
        'LinearSvc': LinearSVC(max_iter=250, penalty='l2', C=0.5),
        'decisionTree': DecisionTreeClassifier(max_depth=4),
        'randomTree': RandomForestClassifier(n_estimators=100, n_jobs=-1, min_samples_leaf=2,
                                             random_state=0),
        'gbdt': GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.1, random_state=0),
        'adaboost': AdaBoostClassifier(n_estimators=300, learning_rate=0.75, random_state=0),
        'extract': ExtraTreesClassifier(n_estimators=250, n_jobs=-1, max_depth=5, random_state=0),
        'gnb': GaussianNB(),
    }
    score,cv_scores,clf=train(data_train,models['randomTree'])
    test(data_test,clf)
    print(score)
    print(cv_scores)

if __name__ == '__main__':
    main()
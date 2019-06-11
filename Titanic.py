# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:41:35 2019
@author: sbtithzy
"""
'''
PassengerID（ID）
Survived(存活与否)
Pclass（客舱等级）
Name（姓名）
Sex（性别）
Age（年龄）
Parch（直系亲友）
SibSp（旁系）
Ticket（票号）
Fare（票价）
Cabin（客舱编号）
Embarked（港口编号）
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
def Ticket_Label(s):
    if (s >= 2) & (s <= 5):####4
        return 2
    elif ((s > 5) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0
def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
###导入训练数据和测试数据 
Train_data = pd.read_csv(r'D:/DM/titanic/train.csv')
Test_data = pd.read_csv(r'D:/DM/titanic/test.csv')
PassengerId = Test_data['PassengerId']
###整合数据
All_data = pd.concat([Train_data,Test_data],ignore_index = 'True')
####特征工程
##1根据名字
All_data['Title'] = All_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
All_data['Title'] = All_data['Title'].map(Title_Dict)
##2根据家庭大小
All_data['FamilySize']=All_data['SibSp']+All_data['Parch']+1###加1表示自己
sns.barplot(x="FamilySize", y="Survived", data=All_data)
####根据家庭大小分为三类 
All_data['FamilyLabel']=All_data['FamilySize'].apply(Fam_label)
sns.barplot(x="FamilyLabel", y="Survived", data=All_data)
##3根据船舱等级
All_data['Cabin'] = All_data['Cabin'].fillna('Unknown')###缺失值填充 
All_data['Deck']=All_data['Cabin'].str.get(0)###新增一个label,取船舱的首字母 
sns.barplot(x="Deck", y="Survived", data=All_data)
##4根据船票
Ticket_Count = dict(All_data['Ticket'].value_counts())
All_data['TicketGroup'] = All_data['Ticket'].apply(lambda x:Ticket_Count[x])
sns.barplot(x='TicketGroup', y='Survived', data=All_data)
#####
All_data['TicketGroup'] = All_data['TicketGroup'].apply(Ticket_Label)
##########数据清洗
###缺失值填充
age_df = All_data[['Age', 'Pclass','Sex','Title']]
age_df=pd.get_dummies(age_df)
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
y = known_age[:, 0]
X = known_age[:, 1:]
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y)
predictedAges = rfr.predict(unknown_age[:, 1::])
All_data.loc[ (All_data.Age.isnull()), 'Age' ] = predictedAges
###
All_data['Embarked'] = All_data['Embarked'].fillna('C')
###
fare=All_data[(All_data['Embarked'] == "S") & (All_data['Pclass'] == 3)].Fare.median()
All_data['Fare']=All_data['Fare'].fillna(fare)
#####同组识别
All_data['Surname']=All_data['Name'].apply(lambda x:x.split(',')[0].strip())
Surname_Count = dict(All_data['Surname'].value_counts())
All_data['FamilyGroup'] = All_data['Surname'].apply(lambda x:Surname_Count[x])
Female_Child_Group=All_data.loc[(All_data['FamilyGroup']>=2) & ((All_data['Age']<=12) | (All_data['Sex']=='female'))]
Male_Adult_Group=All_data.loc[(All_data['FamilyGroup']>=2) & (All_data['Age']>12) & (All_data['Sex']=='male')]
Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns=['GroupCount']
##Female_Child
Male_Adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns=['GroupCount']
##Male_Adult
Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
####
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
#####
Train_data=All_data.loc[All_data['Survived'].notnull()]
Test_data=All_data.loc[All_data['Survived'].isnull()]
Test_data.loc[(Test_data['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'
Test_data.loc[(Test_data['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
Test_data.loc[(Test_data['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'
Test_data.loc[(Test_data['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'
Test_data.loc[(Test_data['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
Test_data.loc[(Test_data['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'
######划分测试集和训练集
All_data=pd.concat([Train_data, Test_data])
All_data=All_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
All_data=pd.get_dummies(All_data)
Train_data=All_data[All_data['Survived'].notnull()]
Test_data=All_data[All_data['Survived'].isnull()].drop('Survived',axis=1)
X = Train_data.as_matrix()[:,1:]
y = Train_data.as_matrix()[:,0]
#######网格搜索训练
###参数优化
pipe=Pipeline([('select',SelectKBest(k=20)), 
               ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])

param_test = {'classify__n_estimators':list(range(10,60,2)), 
              'classify__max_depth':list(range(3,60,3))}
gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, 
                       scoring='roc_auc', cv=10)
gsearch.fit(X,y)
#print(gsearch.best_params_, gsearch.best_score_)###输出参数
#####训练模型
from sklearn.pipeline import make_pipeline
select = SelectKBest(k = 20)
clf = RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 42,#26
                                  max_depth = 6, 
                                  max_features = 'sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)
####交叉验证
cv_score = cross_val_score(pipeline, X, y, cv= 10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))
#####预测
predictions = pipeline.predict(Test_data)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv(r'D:/DM/titanic/submission.csv', index=False)

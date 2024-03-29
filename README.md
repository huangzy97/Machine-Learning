# Machine-Learning
***机器学习***
## part 1  *监督学习*   
### **1.1 分类和回归**  
[Kaggle泰坦尼克比赛](https://github.com/huangzy97/Titanic/edit/master/Titanic.py)   
[![image](https://github.com/huangzy97/lib/blob/master/timg.png)](https://www.kaggle.com/c/titanic)    
#### *1.1.1 KNN(K近邻算法)*  
    K近邻算法的原理：存在一个样本的数据集合(称为训练样本)，并且每个样本都存在标签(理解为分类标签)，这样我们就可以知道样本的每个数据的分类，在输入新的数据后(不附带标签),将新的数据的特征和样本的数据特征进行比较，然后算法提取出样本最相似数据的标签。一般情况下，选择和样本数据集中前K个最为接近的数据，这就是K近邻算法中K的值(K一般为小于20的整数)，最后选择K个中出现最多次数的分类标签作为新的数据分类标签。  
**k-近邻算法步骤如下：**  
1. 计算已知类别数据集中的点与当前点之间的距离；
2. 按照距离递增次序排序；
3. 选取与当前点距离最小的k个点；
4. 确定前k个点所在类别的出现频率；
5. 返回前k个点所出现频率最高的类别作为当前点的预测分类。 
#### *1.1.2 Logistics Regression(逻辑回归算法)*  
    逻辑回归算法的原理：本质上上一种二分类算法，它是利用Sigmoid函数将输出值限制在[0,1]的特性，Logistics Regression进行分类的主要思想是根据现有的数据集对分类边界建立回归公式，然后进行判断，本质上是基于[0,1]的概率进行分类。
*Sigmoid_Function*
![image](https://github.com/huangzy97/lib/blob/master/sigmoid.jpg)  
#### *1.1.3 SVM(支持向量机)*  
##### *1.1.3.1 线性SVM(支持向量机)*  
![image](https://github.com/huangzy97/lib/blob/master/liner_SVM.png)  
##### *1.1.3.2 非线性SVM(支持向量机)*  
![image](https://github.com/huangzy97/lib/blob/master/fei_SVM.png)    
#### *1.1.4 Decision Tree(决策树)*  
##### *1.1.4.1 ID3算法*  
   ID3算法的原理：核心是在决策树各个结点上对应**信息增益**准则选择特征，递归地构建决策树。  
##### *1.1.4.2 C4.5算法*     
   C4.5算法的原理： C4.5是ID3的一个改进算法，最大的区别是C4.5算法用**信息增益比**来选择特征。  
##### *1.1.4.3 CART算法*  
   CART算法的原理：CART算法全称是Classification And Regression Tree，采用的是**Gini指数**（纯净度）来选择特征。  
#### *1.1.5 Random Forest(随机森林)*  
    随机森林算法的原理：随机森林算法是基于决策树算法的集成算法，决策树算法是通过训练集生成一个模型，随机森林是从训练集中随机选取样本(可以抽到相同的样本)，然后生成很多课决策树(构成森林的说法)，然后通过投票的方式进行分类。
## part 2  *无监督学习*  


# Machine-Learning
***机器学习***
## part 1  *监督学习*   
### **1.1 分类**  
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
     Logistics Regression算法的原理：本质上上一种二分类算法，它是利用Sigmoid函数将输出值限制在[0,1]的特性，Logistics Regression进行分类的主要思想是根据现有的数据集对分类边界建立回归公式，然后进行判断，本质上是基于[0,1]的概率进行分类。
*Sigmoid_Function*
![image](https://github.com/huangzy97/lib/blob/master/sigmoid.jpg)
### **1.2 回归**    
## part 2  *无监督学习*  


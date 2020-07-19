# Decision Tree — A Step by Step  _CART (Classification And Regression Tree)_
## **1. Introduction**
Decision Trees is the non-parametric supervised learning approach, and can be applied to both regression and classification problems. In keeping with the tree analogy, decision trees implement a sequential decision process. Starting from the root node, a feature is evaluated and one of the two nodes (branches) is selected, Each node in the tree is basically a decision rule. This procedure is repeated until a final leaf is reached, which normally represents the target. Decision trees are also attractive models if we care about interpretability.

## **2. Various Decision Tree Algorithms**

There are algorithms for creating decision trees :

-   [**ID3**](https://en.wikipedia.org/wiki/ID3_algorithm)  **(Iterative Dichotomiser 3)**  
-   **C4.5**
-   **C5.0**  
-   **CART (Classification and Regression trees)**  is very similar to C4.5, but it differs in that it supports numerical target variables (regression) and does not compute rule sets. CART constructs binary trees using the feature and threshold that yield the largest information gain at each node  [[1]](https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart).

## **3. Decision Tree Terminology**

In keeping with the tree analogy, the terminology was adopted from the terminology of the tree  [[2]](https://gdcoder.com/decision-tree-regressor-explained-in-depth/).
![Image for post](https://miro.medium.com/max/775/1*8hvhzPjR1nU52chzu9FDWg.png)

-   **Root node**  : is the first node in decision trees. It represents entire population or sample and this further gets divided into two or more homogeneous sets.
-   **Splitting**  : is a process of dividing node into two or more sub-nodes, starting from the root node
-   **Node**  : splitting results from the root node into sub-nodes and splitting sub-nodes into further sub-nodes
-   **Leaf or terminal node**  : end of a node, since node cannot be split anymore
-   **Pruning**  : When we remove sub-nodes of a decision node, this process is called pruning. You can say opposite process of splitting. The aim is reducing complexity for improved predictive accuracy and to avoid overfitting
-   **Branch / Sub-Tree**  : A subsection of the entire tree is called branch or sub-tree.
-   **Parent and Child Node**: A node, which is divided into sub-nodes is called parent node of sub-nodes whereas sub-nodes are the child of parent node.
-## **4. Decision Tree Intuition**

### Advantages

1.  **Easy to Understand**: Decision tree output is very easy to understand even for people from non-analytical background. It does not require any statistical knowledge to read and interpret them. Its graphical representation is very intuitive and users can easily relate their hypothesis.
2.  **Useful in Data exploration:** Decision tree is one of the fastest way to identify most significant variables and relation between two or more variables. With the help of decision trees, we can create new variables / features that has better power to predict target variable. You can refer article ([Trick to enhance power of regression model](https://www.analyticsvidhya.com/blog/2013/10/trick-enhance-power-regression-model-2/ "Trick to enhance power of Regression model")) for one such trick. It can also be used in data exploration stage. For example, we are working on a problem where we have information available in hundreds of variables, there decision tree will help to identify most significant variable.
3.  **Less data cleaning required:** It requires less data cleaning compared to some other modeling techniques. It is not influenced by outliers and missing values to a fair degree.
4.  **Data type is not a constraint:** It can handle both numerical and categorical variables.
5.  **Non Parametric Method:** Decision tree is considered to be a non-parametric method. This means that decision trees have no assumptions about the space distribution and the classifier structure.
### Disadvantages

1.  **Over fitting:** Over fitting is one of the most practical difficulty for decision tree models. This problem gets solved by setting constraints on model parameters and pruning (discussed in detailed below).
2.  **Not fit for continuous variables**: While working with continuous numerical variables, decision tree looses information when it categorizes variables in different categories.
## 4. Regression Trees vs Classification Trees
Both the trees work almost similar to each other, let’s look at the primary differences & similarity between classification and regression trees:

1.  Regression trees are used when dependent variable is continuous. Classification trees are used when dependent variable is categorical.
2.  In case of regression tree, the value obtained by terminal nodes in the training data is the mean response of observation falling in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mean value.
3.  In case of classification tree, the value (class) obtained by terminal node in the training data is the mode of observations falling in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mode value.
4.  Both the trees divide the predictor space (independent variables) into distinct and non-overlapping regions. For the sake of simplicity, you can think of these regions as high dimensional boxes or boxes.
5.  Both the trees follow a top-down greedy approach known as recursive binary splitting. We call it as ‘top-down’ because it begins from the top of tree when all the observations are available in a single region and successively splits the predictor space into two new branches down the tree. It is known as ‘greedy’ because, the algorithm cares (looks for best variable available) about only the current split, and not about future splits which will lead to a better tree.
6.  This splitting process is continued until a user defined stopping criteria is reached. For example: we can tell the the algorithm to stop once the number of observations per node becomes less than 50.
7.  In both the cases, the splitting process results in fully grown trees until the stopping criteria is reached.  But, the fully grown tree is likely to overfit data, leading to poor accuracy on unseen data. This bring ‘pruning’. Pruning is one of the technique used tackle overfitting. We’ll learn more about it in following section.
## **5. Splitting in Decision Trees**

In order to split the nodes at the most informative features using the decision algorithm, we start at the tree root and split the data on the feature that results in the largest information gain (IG). Here, the objective function is to maximize the information gain (IG) at each split, which we define as follows:
![Image for post](https://miro.medium.com/max/501/1*qpUAC2VxA1KFMAyrEINREQ.png)

_f_  is the feature to perform the split,  _Dp_  and  _Dj_  are data set of the parent,  _j_-th child node,  _I_  is our impurity measure,  _Np_  is the total number of samples at the parent node, and  _Nj_  is the number of samples in the  _j_-th child node.

As we can see, the information gain is simply the difference between the impurity of the parent node and the sum of the child node impurities — the lower the impurity of the child nodes, the larger the information gain. however, for simplicity and to reduce the combinatorial search space, most libraries (including scikit-learn) implement binary decision trees. This means that each parent node is split into two child nodes,  _D-left_ and  _D-right._
![Image for post](https://miro.medium.com/max/652/1*R_GDx8NhSZ_p27EN8Wh6EQ.png)

impurity measure implements binary decisions trees and the three impurity measures or splitting criteria that are commonly used in binary decision trees are  _Gini impurity (IG)_,  _entropy (IH)_, and _misclassification error (IE)_.
## **5.1 Gini Impurity** 
## **5.1.1 _Step by Step  CART work in Classification_**
_Used by the CART (classification and regression tree) algorithm for classification trees_, _Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset._
Mathematically, we can write Gini Impurity as following
![Image for post](https://miro.medium.com/max/354/1*V5NKEredkTuDnoQPyjmyOw.png)

where  _j_ is the number of classes present in the node and  _p_ is the distribution of the class in the node.

Simple simulation with  [Heart Disease Data set](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)  with 303 rows and has 13 attributes. Target consist 138 value 0 and 165 value 1
![Image for post](https://miro.medium.com/max/505/0*UxGWvrRifh3dqWOQ)
In order to build a decision tree from the dataset and to determine which separation is best, we need a way to measure and compare Gini Impurity in each attribute. The lowest Gini Impurity value on the first iteration will be the Root Node. we can write equation 3 as :
![Image for post](https://miro.medium.com/max/795/1*mRptCs22opsiVSglWd5yGg.png)
In this simulation, only use the sex, fbs (fasting blood sugar), exang (exercise induced angina), and target attributes.

**Measure Gini Impurity in Sex**
![Image for post](https://miro.medium.com/max/523/1*GOdkYC7ahc7OvXoGS4PBBw.png)

![Image for post](https://miro.medium.com/max/982/1*weJz1ySCM1zCdUXXhnDnfQ.png)

**Measure Gini Impurity in Fbs (fasting blood sugar)**
![Image for post](https://miro.medium.com/max/523/1*maoBf7OGfIdz-M_6tocROw.png)

![Image for post](https://miro.medium.com/max/60/1*uitcZpl9-DCOAdcI5kzQDw.png?q=20)

![Image for post](https://miro.medium.com/max/1025/1*uitcZpl9-DCOAdcI5kzQDw.png)

**Measure Gini Impurity in Exang (exercise induced angina)**
![Image for post](https://miro.medium.com/max/523/1*uv5yAlR-7zjpCdj3d1iNjA.png)

![Image for post](https://miro.medium.com/max/60/1*ErP1iQA29UeKEYtqGY_5dA.png?q=20)

![Image for post](https://miro.medium.com/max/811/1*ErP1iQA29UeKEYtqGY_5dA.png)

**Fbs (fasting blood sugar) has the lowest Gini Impurity, so well use it at the Root Node**
As we know, we have Fbs as Root Node, when we divide all of the patients using Fbs (fasting blood sugar), we end up with “Impure” leaf nodes. Each leaf contained with and without heart disease.
![Image for post](https://miro.medium.com/max/654/0*WG30IbLeSBfqfo2H)

we need to figure how well Sex and Exang separate these patient in left node of Fbs
![Image for post](https://miro.medium.com/max/921/0*qhDveP9TvqqCn_Uy)

**Exang (exercise induced angina) has the lowest Gini Impurity, we will use it at this node to separate patients.**
![Image for post](https://miro.medium.com/max/366/0*4j3X8XYg7gdY50Ld)

In the left node of Exang (exercise induced angina), how well it separate these 49 patients (24 with heart disease and 25 without heart disease. Since only the attribute sex is left, we put sex attribute in the left node of Exang
![Image for post](https://miro.medium.com/max/1197/0*CqUhQD6zzZgBg1bS)

As we can see, we have final leaf nodes on this branch, but why is the leaf node circled including the final node?

Note : the leaf node circled, 89% don’t have heart diseases

Do these new leaves separate patients better than what we had before ?

In order to answer those question, we must compare Gini Impurity using attribute sex and Gini Impurity before using attribute sex to separate patients.
![Image for post](https://miro.medium.com/max/1242/0*0Fut-9j0qxgXShRa)

The Gini Impurity before using sex to separate patients is lowest, so we don’t separate this node using Sex. The final leaf node on this branch of tree
![Image for post](https://miro.medium.com/max/457/0*ttXZLxx9OC75BAc8)

Do the same thing on the right branch, so the end result of a tree in this case is
![Image for post](https://miro.medium.com/max/830/0*HPE2L-ik2iHKsoNJ)

**Main point when process the splitting of the dataset**

> 1. calculate all of the Gini impurity score
> 2. compare the Gini impurity score, after n before using new attribute to separate data. If the node itself has the lowest score, than there is no point in separating the data
> 3. If separating the data result in an improvement, than pick the separation with the lowest impurity score
# Bonus

## **5.1.1.1 How to calculate Gini Impurity in continuous data?**

such as weight which is one of the attributes to determine heart disease, for example we have weight attribute
![Image for post](https://miro.medium.com/max/321/0*zG_vM-YBOzP_Fcm6)

**Step 1 : Order data by ascending**
![Image for post](https://miro.medium.com/max/375/0*cUK8HXBDgD0TovYN)

**Step 2 : Calculate the average weight**
![Image for post](https://miro.medium.com/max/561/0*UDVMwgSRWS4gwl7q)

**Step 3 : Calculate Gini Impurity values for each average weight**
![Image for post](https://miro.medium.com/max/1600/0*9n29hkwLQyOeg9OT)

The lowest Gini Impurity is  **Weight < 205,** this is the cutoff and impurity value if used when we compare with another attribute

## **5.1.1.2How to calculate Gini Impurity in categorical data?**

we have a favorite color attribute to determine a person’s gender
![Image for post](https://miro.medium.com/max/300/0*EiA4It23nkFcq-aS)

In order to know Gini Impurity this attribute, calculate an impurity score for each one as well as each possible combination
![Image for post](https://miro.medium.com/max/886/0*l3nAu8xRF4qmcGuh)

Now, we have possible combination and we find out the lowest Gini Impurity to determine cutoff and impurity value
## **5.1.2 CART Work in Regression with one predictor**
CART in classification cases uses Gini Impurity in the process of splitting the dataset into a decision tree. On the other hand CART in regression cases uses least squares, intuitively splits are chosen to minimize the  **residual sum of squares** between the observation and the mean in each node. Mathematically, we can write residual as follow
![Image for post](https://miro.medium.com/max/226/1*dfRaEKhUfa0NonQwZbS4DQ.png)

Mathematically, we can write  **RSS (residual sum of squares)** as follow
![Image for post](https://miro.medium.com/max/360/1*VJd4g9In8DTnOuV49QJ2yg.png)

## **In order to find out the “best” split, we must minimize the RSS**

## **5.1.2.1 Intuition**

This simulation uses a “dummy” dataset  as follow
![Image for post](https://miro.medium.com/max/1001/1*HsDCChP7tw-V9fA1CpqyHg.png)

The decision tree as follow
![Image for post](https://miro.medium.com/max/1362/1*7AY2-g45h8eh6p2b-bU3Rg.png)

## **How does CART process the splitting of the dataset (predictor =1)**
CART in classification cases uses Gini Impurity in the process of splitting the dataset into a decision tree. On the other hand CART in regression cases uses least squares, intuitively splits are chosen to minimize the  **residual sum of squares** between the observation and the mean in each node. Mathematically, we can write residual as follow
![Image for post](https://miro.medium.com/max/226/1*dfRaEKhUfa0NonQwZbS4DQ.png)

Mathematically, we can write  **RSS (residual sum of squares)** as follow
![Image for post](https://miro.medium.com/max/360/1*VJd4g9In8DTnOuV49QJ2yg.png)

## **In order to find out the “best” split, we must minimize the RSS**

## **5.1.2.2 Intuition**

This simulation uses a “dummy” dataset  as follow
![Image for post](https://miro.medium.com/max/1001/1*HsDCChP7tw-V9fA1CpqyHg.png)

The decision tree as follow
![Image for post](https://miro.medium.com/max/1362/1*7AY2-g45h8eh6p2b-bU3Rg.png)

## **5.1.3 How does CART process the splitting of the dataset (predictor =1)**
As mentioned before,  **In order to find out the “best” split, we must minimize the RSS.** first, we calculate  **RSS**  by split into two regions, start with index 0

**Start within index 0**
![Image for post](https://miro.medium.com/max/1316/1*955o8f61OFgnlMZSTVsw-A.png)

The data already split into two regions, we add up the squared residual for every index data. furthermore we calculate  **RSS**  each node using equation 2.0
![Image for post](https://miro.medium.com/max/1664/1*9CVl4sc9L95MHTt4uxlq0w.png)

## **Start within index 1**

calculate  **RSS**  by split into two regions within index 1
![Image for post](https://miro.medium.com/max/1316/1*ca_81--21ZGPo6Lvonnwig.png)

after the data is divided into two regions then calculate  **RSS**  each node using equation 2.0
![Image for post](https://miro.medium.com/max/1677/1*sZLI2-gmTzvR7sxAYMSOzg.png)

## Start within index 2

calculate  **RSS**  by split into two regions within index 2
![Image for post](https://miro.medium.com/max/1216/1*LiVz3Bp8FrtPh36tVq45tA.png)

calculate  **RSS**  each node
![Image for post](https://miro.medium.com/max/1591/1*38K1pGNnSBWF5RmUucIEXA.png)

This process continues until the calculation of RSS in the last index

**Last Index**
![Image for post](https://miro.medium.com/max/1286/1*IsuxU0Rq-lwlrxPVFWZLjw.png)

Price with threshold 19 has a smallest RSS, in R1 there are 10 data within price < 19, so we’ll split the data in R1. In order to avoid overfitting, we define the minimum data for each region >= 6. If the region has less than 6 data, the split process in that region stops.

Split the data with threshold 19
![Image for post](https://miro.medium.com/max/899/1*Zf_LRyCbUskdLW5PEe23Ng.png)

calculate RSS in R1, the process in this section is the same as the previous process, only done for R1
![Image for post](https://miro.medium.com/max/1477/1*dGJQ4TR3VcrP1WE-yD21xg.png)

Do the same thing on the right branch, so the end result of a tree in this case is

## **5.1.4 How does CART process the splitting of the dataset (predictor > 1)**

This simulation uses a  **dummy data** as following
![Image for post](https://miro.medium.com/max/315/1*PK4-6diJoEvBEEw2LBLeYw.png)

Find out the minimum RSS each predictor

**Price with RSS = 3873.79**
![Image for post](https://miro.medium.com/max/1286/1*IsuxU0Rq-lwlrxPVFWZLjw.png)

**Cleaning fee with RSS = 64214.8**
![Image for post](https://miro.medium.com/max/1286/1*Zyhe1i-Cgoxggtvebn2HqA.png)

There is only one threshold in License, 1 or 0. So we use that threshold to calculate RSS.  **License with RSS = 11658.5**
![Image for post](https://miro.medium.com/max/1281/1*RvqsbVz6arOvScn9bi2z7A.png)

**We already have RSS every predictor, compare RSS for each predictor, and find the lowest RSS value. If we analyze, License has the lowest value so it becomes root.**

## **5.2 Entropy**

Used by the ID3, C4.5 and C5.0 tree-generation algorithms. Information gain is based on the concept of entropy, the entropy measure is defined as
![Image for post](https://miro.medium.com/max/315/1*coWr5c4M7IQVEo9OzJanvw.png)

where  _j_ is the number of classes present in the node and  _p_ is the distribution of the class in the node.

In the same case and same  [data set](https://archive.ics.uci.edu/ml/datasets/Heart+Disease), we need a way to measure and compare Entropy in each attribute. The highest Entropy value on the first iteration will be the Root Node.

We need calculate entropy in Target attribute first
![Image for post](https://miro.medium.com/max/700/1*KFgTicw0qBw60p1juuaRzQ.png)

**How to measure Entropy in Sex attribute**
![Image for post](https://miro.medium.com/max/523/1*2ndE048b-mZsOW7jGhQm2w.png)

Entropy — Sex = 0
![Image for post](https://miro.medium.com/max/652/1*Jrc501ITDFcoQtrE533Pjw.png)

Entropy — Sex = 1
![Image for post](https://miro.medium.com/max/617/1*hABWmggHDGxCYao19t_vDA.png)

Now that we have measured the Entropy for both leaf nodes. We take the weight average again to calculate the total entropy value.

Entropy — Sex
![Image for post](https://miro.medium.com/max/777/1*Kbv0juuzWksRIKBfNr2edQ.png)

**How to measure Entropy in Fbs attribute**
![Image for post](https://miro.medium.com/max/523/1*nd8JBT_sNcRmfoPmE7qu2g.png)

Entropy — Fbs = 0
![Image for post](https://miro.medium.com/max/659/1*nb1zrw6St72BWPgiV9Ir0w.png)

Entropy — Fbs = 1
![Image for post](https://miro.medium.com/max/651/1*g7vBJnaaqH9eZlevjizY1Q.png)

Entropy — Fbs
![Image for post](https://miro.medium.com/max/783/1*yy2UDKaOIkjYpSheeC5LtA.png)

**How to measure Entropy in Exang attribute**
![Image for post](https://miro.medium.com/max/523/1*imlmEAq5Pff0TQwbT1qPEw.png)

Entropy — Exang = 0
![Image for post](https://miro.medium.com/max/657/1*4aoH9A5C-slBy0dfyY8Nwg.png)

Entropy — Exang = 1
![Image for post](https://miro.medium.com/max/679/1*igHmyuWwQzbB0k7ui8Vy9A.png)

Entropy — Exang
![Image for post](https://miro.medium.com/max/831/1*1kFkZKlMtzjuI2m2XY0NQg.png)

**Fbs (fasting blood sugar) has the highest gini impurity, so we will use it at the Root Node, Precisely the same results we got from Gini Impurity.**
## **5.3 Misclassification Impurity**

Another impurity measure is the misclassification impurity , Mathematically, we can write misclassification impurity as following
![Image for post](https://miro.medium.com/max/348/1*z7vOj6jpq8ZOmXfA0ISW3w.png)

In terms of quality performance, this index is not the best choice because it’s not particularly sensitive to different probability distributions (which can easily drive the selection to a subdivision using Gini or entropy).

## **5.4 Chi-Square**
It is an algorithm to find out the statistical significance between the differences between sub-nodes and parent node. We measure it by sum of squares of standardized differences between observed and expected frequencies of target variable.

1.  It works with categorical target variable “Success” or “Failure”.
2.  It can perform two or more splits.
3.  Higher the value of Chi-Square higher the statistical significance of differences between sub-node and Parent node.
4.  Chi-Square of each node is calculated using formula,
5.  Chi-square = ((Actual – Expected)^2 / Expected)^1/2
6.  It generates tree called CHAID (Chi-square Automatic Interaction Detector)

**Steps to Calculate Chi-square for a split:**

1.  Calculate Chi-square for individual node by calculating the deviation for Success and Failure both
2.  Calculated Chi-square of Split using Sum of all Chi-square of success and Failure of each node of the split

**Example:**![Decision Tree, Algorithm, Gini Index](https://www.analyticsvidhya.com/wp-content/uploads/2015/01/Decision_Tree_Algorithm1.png)
**Split on Gender:**

1.  First we are populating for node Female, Populate the actual value for “**Play Cricket”**  and **“Not Play Cricket”**, here these are 2 and 8 respectively.
2.  Calculate expected value for “**Play Cricket”**  and “**Not Play Cricket”**, here it would be 5 for both because parent node has probability of 50% and we have applied same probability on Female count(10).
3.  Calculate deviations by using formula, Actual – Expected. It is for “**Play Cricket”**  (2 – 5 = -3) and for “**Not play cricket”**  ( 8 – 5 = 3).
4.  Calculate Chi-square of node for “**Play Cricket**” and “**Not Play Cricket**” using formula with formula,  **= ((Actual – Expected)^2 / Expected)^1/2**. You can refer below table for calculation.
5.  Follow similar steps for calculating Chi-square value for Male node.
6.  Now add all Chi-square values to calculate Chi-square for split Gender.
![Decision Tree, Chi-Square](https://www.analyticsvidhya.com/wp-content/uploads/2015/01/Decision_Tree_Chi_Square1.png)
**Split on Class:**

Perform similar steps of calculation for split on Class and you will come up with below table.

![Decision Tree, Chi-Square](https://www.analyticsvidhya.com/wp-content/uploads/2015/01/Decision_Tree_Chi_Square_2.png)
Above, you can see that Chi-square also identify the Gender split is more significant compare to Class.

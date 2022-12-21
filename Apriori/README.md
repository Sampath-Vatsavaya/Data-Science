# Generating Frequent Itemsets / Strong Association Rules using 
# Apriori Algorithm




**Apriori algorithm** is for finding frequent itemsets in a dataset for boolean association rule.
Name of the algorithm is Apriori because it uses prior knowledge of frequent itemset
properties.

We apply an iterative approach or level-wise search where k-frequent itemsets are
used to find k+1 itemsets. To improve the efficiency of level-wise generation of frequent
itemsets, an important property is used called Apriori property which helps by reducing the
search space.

## Step-1: Importing the libraries


```python

import pandas as pd
import numpy as np
from apyori import apriori
```

## Step-2: Loading the Data and Preprocessing


```python
df =pd.read_csv('Market_Basket_Optimisation.csv',header=None)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>shrimp</td>
      <td>almonds</td>
      <td>avocado</td>
      <td>vegetables mix</td>
      <td>green grapes</td>
      <td>whole weat flour</td>
      <td>yams</td>
      <td>cottage cheese</td>
      <td>energy drink</td>
      <td>tomato juice</td>
      <td>low fat yogurt</td>
      <td>green tea</td>
      <td>honey</td>
      <td>salad</td>
      <td>mineral water</td>
      <td>salmon</td>
      <td>antioxydant juice</td>
      <td>frozen smoothie</td>
      <td>spinach</td>
      <td>olive oil</td>
    </tr>
    <tr>
      <th>1</th>
      <td>burgers</td>
      <td>meatballs</td>
      <td>eggs</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chutney</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>turkey</td>
      <td>avocado</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mineral water</td>
      <td>milk</td>
      <td>energy bar</td>
      <td>whole wheat rice</td>
      <td>green tea</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(0,inplace=True)
transactions=[]
for i in range(0,len(df)):
    transactions.append([str(df.values[i,j]) for j in range(0,20) if str(df.values[i,j])!='0'])
transactions[0]

```




    ['shrimp',
     'almonds',
     'avocado',
     'vegetables mix',
     'green grapes',
     'whole weat flour',
     'yams',
     'cottage cheese',
     'energy drink',
     'tomato juice',
     'low fat yogurt',
     'green tea',
     'honey',
     'salad',
     'mineral water',
     'salmon',
     'antioxydant juice',
     'frozen smoothie',
     'spinach',
     'olive oil']



## Step-3: Applying Apriori


```python
rules = apriori(transactions,min_support=0.003,min_confidance=0.2,min_lift=3,min_length=2)
rules
```




    <generator object apriori at 0x0000023F21269A50>




```python
Results = list(rules)
Results
```




    [RelationRecord(items=frozenset({'brownies', 'cottage cheese'}), support=0.0034662045060658577, ordered_statistics=[OrderedStatistic(items_base=frozenset({'brownies'}), items_add=frozenset({'cottage cheese'}), confidence=0.10276679841897232, lift=3.225329518580382), OrderedStatistic(items_base=frozenset({'cottage cheese'}), items_add=frozenset({'brownies'}), confidence=0.10878661087866107, lift=3.2253295185803816)]),
     RelationRecord(items=frozenset({'chicken', 'light cream'}), support=0.004532728969470737, ordered_statistics=[OrderedStatistic(items_base=frozenset({'chicken'}), items_add=frozenset({'light cream'}), confidence=0.07555555555555556, lift=4.843950617283951), OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'chicken'}), confidence=0.29059829059829057, lift=4.84395061728395)]),
     RelationRecord(items=frozenset({'escalope', 'mushroom cream sauce'}), support=0.005732568990801226, ordered_statistics=[OrderedStatistic(items_base=frozenset({'escalope'}), items_add=frozenset({'mushroom cream sauce'}), confidence=0.0722689075630252, lift=3.7908326967150496), OrderedStatistic(items_base=frozenset({'mushroom cream sauce'}), items_add=frozenset({'escalope'}), confidence=0.3006993006993007, lift=3.790832696715049)]),
     RelationRecord(items=frozenset({'pasta', 'escalope'}), support=0.005865884548726837, ordered_statistics=[OrderedStatistic(items_base=frozenset({'escalope'}), items_add=frozenset({'pasta'}), confidence=0.07394957983193277, lift=4.700811850163794), OrderedStatistic(items_base=frozenset({'pasta'}), items_add=frozenset({'escalope'}), confidence=0.3728813559322034, lift=4.700811850163794)]),
     RelationRecord(items=frozenset({'fresh bread', 'tomato juice'}), support=0.004266097853619517, ordered_statistics=[OrderedStatistic(items_base=frozenset({'fresh bread'}), items_add=frozenset({'tomato juice'}), confidence=0.09907120743034055, lift=3.2593558198902826), OrderedStatistic(items_base=frozenset({'tomato juice'}), items_add=frozenset({'fresh bread'}), confidence=0.14035087719298245, lift=3.2593558198902826)]),
     RelationRecord(items=frozenset({'fresh tuna', 'honey'}), support=0.003999466737768298, ordered_statistics=[OrderedStatistic(items_base=frozenset({'fresh tuna'}), items_add=frozenset({'honey'}), confidence=0.17964071856287428, lift=3.7850703088205613), OrderedStatistic(items_base=frozenset({'honey'}), items_add=frozenset({'fresh tuna'}), confidence=0.08426966292134831, lift=3.7850703088205613)]),
     RelationRecord(items=frozenset({'fromage blanc', 'honey'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'fromage blanc'}), items_add=frozenset({'honey'}), confidence=0.2450980392156863, lift=5.164270764485569), OrderedStatistic(items_base=frozenset({'honey'}), items_add=frozenset({'fromage blanc'}), confidence=0.0702247191011236, lift=5.16427076448557)]),
     RelationRecord(items=frozenset({'ground beef', 'herb & pepper'}), support=0.015997866951073192, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'herb & pepper'}), confidence=0.1628222523744912, lift=3.291993841134928), OrderedStatistic(items_base=frozenset({'herb & pepper'}), items_add=frozenset({'ground beef'}), confidence=0.3234501347708895, lift=3.2919938411349285)]),
     RelationRecord(items=frozenset({'ground beef', 'tomato sauce'}), support=0.005332622317024397, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'tomato sauce'}), confidence=0.054274084124830396, lift=3.840659481324083), OrderedStatistic(items_base=frozenset({'tomato sauce'}), items_add=frozenset({'ground beef'}), confidence=0.3773584905660377, lift=3.840659481324083)]),
     RelationRecord(items=frozenset({'light cream', 'olive oil'}), support=0.003199573390214638, ordered_statistics=[OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'olive oil'}), confidence=0.20512820512820515, lift=3.1147098515519573), OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'light cream'}), confidence=0.048582995951417005, lift=3.114709851551957)]),
     RelationRecord(items=frozenset({'whole wheat pasta', 'olive oil'}), support=0.007998933475536596, ordered_statistics=[OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'whole wheat pasta'}), confidence=0.12145748987854252, lift=4.1224100976422955), OrderedStatistic(items_base=frozenset({'whole wheat pasta'}), items_add=frozenset({'olive oil'}), confidence=0.2714932126696833, lift=4.122410097642296)]),
     RelationRecord(items=frozenset({'pasta', 'shrimp'}), support=0.005065991201173177, ordered_statistics=[OrderedStatistic(items_base=frozenset({'pasta'}), items_add=frozenset({'shrimp'}), confidence=0.3220338983050847, lift=4.506672147735896), OrderedStatistic(items_base=frozenset({'shrimp'}), items_add=frozenset({'pasta'}), confidence=0.0708955223880597, lift=4.506672147735896)]),
     RelationRecord(items=frozenset({'spaghetti', 'avocado', 'milk'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'spaghetti', 'avocado'}), confidence=0.025720164609053502, lift=3.2154492455418384), OrderedStatistic(items_base=frozenset({'spaghetti', 'avocado'}), items_add=frozenset({'milk'}), confidence=0.41666666666666663, lift=3.215449245541838)]),
     RelationRecord(items=frozenset({'burgers', 'cake', 'milk'}), support=0.0037328356219170776, ordered_statistics=[OrderedStatistic(items_base=frozenset({'burgers'}), items_add=frozenset({'cake', 'milk'}), confidence=0.04281345565749235, lift=3.211437308868501), OrderedStatistic(items_base=frozenset({'cake', 'milk'}), items_add=frozenset({'burgers'}), confidence=0.27999999999999997, lift=3.211437308868501)]),
     RelationRecord(items=frozenset({'burgers', 'chocolate', 'turkey'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'burgers'}), items_add=frozenset({'turkey', 'chocolate'}), confidence=0.035168195718654434, lift=3.1034898363014927), OrderedStatistic(items_base=frozenset({'turkey', 'chocolate'}), items_add=frozenset({'burgers'}), confidence=0.27058823529411763, lift=3.1034898363014927)]),
     RelationRecord(items=frozenset({'burgers', 'turkey', 'milk'}), support=0.003199573390214638, ordered_statistics=[OrderedStatistic(items_base=frozenset({'burgers'}), items_add=frozenset({'turkey', 'milk'}), confidence=0.03669724770642202, lift=3.2384241770102538), OrderedStatistic(items_base=frozenset({'turkey', 'milk'}), items_add=frozenset({'burgers'}), confidence=0.2823529411764706, lift=3.2384241770102533)]),
     RelationRecord(items=frozenset({'cake', 'frozen vegetables', 'tomatoes'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'cake', 'tomatoes'}), confidence=0.03216783216783217, lift=3.83001443001443), OrderedStatistic(items_base=frozenset({'tomatoes'}), items_add=frozenset({'cake', 'frozen vegetables'}), confidence=0.04483430799220273, lift=4.367560314928736), OrderedStatistic(items_base=frozenset({'cake', 'frozen vegetables'}), items_add=frozenset({'tomatoes'}), confidence=0.2987012987012987, lift=4.367560314928736), OrderedStatistic(items_base=frozenset({'cake', 'tomatoes'}), items_add=frozenset({'frozen vegetables'}), confidence=0.36507936507936506, lift=3.8300144300144296)]),
     RelationRecord(items=frozenset({'cereals', 'ground beef', 'spaghetti'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'cereals'}), items_add=frozenset({'ground beef', 'spaghetti'}), confidence=0.11917098445595856, lift=3.040481477565119), OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'cereals', 'spaghetti'}), confidence=0.03120759837177748, lift=4.681763907734057), OrderedStatistic(items_base=frozenset({'spaghetti'}), items_add=frozenset({'cereals', 'ground beef'}), confidence=0.01761102603369066, lift=3.885303125844519), OrderedStatistic(items_base=frozenset({'cereals', 'ground beef'}), items_add=frozenset({'spaghetti'}), confidence=0.6764705882352942, lift=3.8853031258445188), OrderedStatistic(items_base=frozenset({'cereals', 'spaghetti'}), items_add=frozenset({'ground beef'}), confidence=0.45999999999999996, lift=4.681763907734057), OrderedStatistic(items_base=frozenset({'ground beef', 'spaghetti'}), items_add=frozenset({'cereals'}), confidence=0.0782312925170068, lift=3.0404814775651197)]),
     RelationRecord(items=frozenset({'chicken', 'ground beef', 'milk'}), support=0.0038661511798426876, ordered_statistics=[OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'chicken', 'ground beef'}), confidence=0.02983539094650206, lift=3.1520460209818584), OrderedStatistic(items_base=frozenset({'chicken', 'ground beef'}), items_add=frozenset({'milk'}), confidence=0.40845070422535207, lift=3.152046020981858)]),
     RelationRecord(items=frozenset({'chicken', 'olive oil', 'milk'}), support=0.0035995200639914677, ordered_statistics=[OrderedStatistic(items_base=frozenset({'chicken'}), items_add=frozenset({'olive oil', 'milk'}), confidence=0.06, lift=3.51609375), OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'chicken', 'olive oil'}), confidence=0.02777777777777778, lift=3.8585390946502063), OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'chicken', 'milk'}), confidence=0.054655870445344125, lift=3.6934566145092456), OrderedStatistic(items_base=frozenset({'chicken', 'milk'}), items_add=frozenset({'olive oil'}), confidence=0.24324324324324323, lift=3.693456614509246), OrderedStatistic(items_base=frozenset({'chicken', 'olive oil'}), items_add=frozenset({'milk'}), confidence=0.5, lift=3.858539094650206), OrderedStatistic(items_base=frozenset({'olive oil', 'milk'}), items_add=frozenset({'chicken'}), confidence=0.2109375, lift=3.51609375)]),
     RelationRecord(items=frozenset({'chicken', 'pancakes', 'milk'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'chicken'}), items_add=frozenset({'pancakes', 'milk'}), confidence=0.051111111111111114, lift=3.091810035842294), OrderedStatistic(items_base=frozenset({'pancakes', 'milk'}), items_add=frozenset({'chicken'}), confidence=0.18548387096774194, lift=3.091810035842294)]),
     RelationRecord(items=frozenset({'chicken', 'olive oil', 'spaghetti'}), support=0.0034662045060658577, ordered_statistics=[OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'chicken', 'spaghetti'}), confidence=0.05263157894736842, lift=3.0603835169318647), OrderedStatistic(items_base=frozenset({'chicken', 'spaghetti'}), items_add=frozenset({'olive oil'}), confidence=0.20155038759689922, lift=3.0603835169318647)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'chocolate', 'shrimp'}), support=0.005332622317024397, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'chocolate', 'shrimp'}), confidence=0.055944055944055944, lift=3.1084175084175087), OrderedStatistic(items_base=frozenset({'shrimp'}), items_add=frozenset({'frozen vegetables', 'chocolate'}), confidence=0.07462686567164178, lift=3.2545123221103784), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'chocolate'}), items_add=frozenset({'shrimp'}), confidence=0.23255813953488375, lift=3.2545123221103784), OrderedStatistic(items_base=frozenset({'chocolate', 'shrimp'}), items_add=frozenset({'frozen vegetables'}), confidence=0.29629629629629634, lift=3.1084175084175087)]),
     RelationRecord(items=frozenset({'ground beef', 'herb & pepper', 'chocolate'}), support=0.003999466737768298, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'herb & pepper', 'chocolate'}), confidence=0.0407055630936228, lift=4.490182775959774), OrderedStatistic(items_base=frozenset({'herb & pepper'}), items_add=frozenset({'ground beef', 'chocolate'}), confidence=0.08086253369272238, lift=3.5060685851393676), OrderedStatistic(items_base=frozenset({'ground beef', 'chocolate'}), items_add=frozenset({'herb & pepper'}), confidence=0.17341040462427748, lift=3.5060685851393676), OrderedStatistic(items_base=frozenset({'herb & pepper', 'chocolate'}), items_add=frozenset({'ground beef'}), confidence=0.4411764705882354, lift=4.4901827759597746)]),
     RelationRecord(items=frozenset({'soup', 'chocolate', 'milk'}), support=0.003999466737768298, ordered_statistics=[OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'soup', 'chocolate'}), confidence=0.030864197530864203, lift=3.0462150747238472), OrderedStatistic(items_base=frozenset({'soup', 'chocolate'}), items_add=frozenset({'milk'}), confidence=0.3947368421052632, lift=3.0462150747238472)]),
     RelationRecord(items=frozenset({'ground beef', 'cooking oil', 'spaghetti'}), support=0.004799360085321957, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'cooking oil', 'spaghetti'}), confidence=0.048846675712347354, lift=3.078982474943844), OrderedStatistic(items_base=frozenset({'spaghetti'}), items_add=frozenset({'ground beef', 'cooking oil'}), confidence=0.027565084226646247, lift=3.2819951870487856), OrderedStatistic(items_base=frozenset({'ground beef', 'cooking oil'}), items_add=frozenset({'spaghetti'}), confidence=0.5714285714285714, lift=3.2819951870487856), OrderedStatistic(items_base=frozenset({'cooking oil', 'spaghetti'}), items_add=frozenset({'ground beef'}), confidence=0.3025210084033613, lift=3.0789824749438446)]),
     RelationRecord(items=frozenset({'eggs', 'ground beef', 'herb & pepper'}), support=0.0041327822956939075, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'eggs', 'herb & pepper'}), confidence=0.04206241519674356, lift=3.3564912381997174), OrderedStatistic(items_base=frozenset({'herb & pepper'}), items_add=frozenset({'eggs', 'ground beef'}), confidence=0.08355795148247978, lift=4.178454627133872), OrderedStatistic(items_base=frozenset({'eggs', 'ground beef'}), items_add=frozenset({'herb & pepper'}), confidence=0.2066666666666667, lift=4.178454627133872), OrderedStatistic(items_base=frozenset({'eggs', 'herb & pepper'}), items_add=frozenset({'ground beef'}), confidence=0.3297872340425532, lift=3.3564912381997174)]),
     RelationRecord(items=frozenset({'eggs', 'spaghetti', 'red wine'}), support=0.0037328356219170776, ordered_statistics=[OrderedStatistic(items_base=frozenset({'red wine'}), items_add=frozenset({'eggs', 'spaghetti'}), confidence=0.13270142180094788, lift=3.6328224997405476), OrderedStatistic(items_base=frozenset({'spaghetti'}), items_add=frozenset({'eggs', 'red wine'}), confidence=0.021439509954058193, lift=3.0342974370828397), OrderedStatistic(items_base=frozenset({'eggs', 'red wine'}), items_add=frozenset({'spaghetti'}), confidence=0.5283018867924528, lift=3.0342974370828397), OrderedStatistic(items_base=frozenset({'eggs', 'spaghetti'}), items_add=frozenset({'red wine'}), confidence=0.10218978102189781, lift=3.632822499740547)]),
     RelationRecord(items=frozenset({'ground beef', 'herb & pepper', 'french fries'}), support=0.003199573390214638, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'herb & pepper', 'french fries'}), confidence=0.03256445047489824, lift=4.697421981004071), OrderedStatistic(items_base=frozenset({'herb & pepper'}), items_add=frozenset({'ground beef', 'french fries'}), confidence=0.0646900269541779, lift=4.665768194070082), OrderedStatistic(items_base=frozenset({'ground beef', 'french fries'}), items_add=frozenset({'herb & pepper'}), confidence=0.23076923076923078, lift=4.665768194070081), OrderedStatistic(items_base=frozenset({'herb & pepper', 'french fries'}), items_add=frozenset({'ground beef'}), confidence=0.46153846153846156, lift=4.697421981004071)]),
     RelationRecord(items=frozenset({'green tea', 'frozen vegetables', 'tomatoes'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'tomatoes'}), items_add=frozenset({'green tea', 'frozen vegetables'}), confidence=0.04873294346978558, lift=3.38468341635983), OrderedStatistic(items_base=frozenset({'green tea', 'frozen vegetables'}), items_add=frozenset({'tomatoes'}), confidence=0.2314814814814815, lift=3.38468341635983)]),
     RelationRecord(items=frozenset({'ground beef', 'frozen vegetables', 'spaghetti'}), support=0.008665511265164644, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'frozen vegetables', 'spaghetti'}), confidence=0.08819538670284939, lift=3.165328208890303), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti'}), items_add=frozenset({'ground beef'}), confidence=0.31100478468899523, lift=3.165328208890303)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'olive oil', 'milk'}), support=0.004799360085321957, ordered_statistics=[OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'frozen vegetables', 'olive oil'}), confidence=0.037037037037037035, lift=3.2684095860566447), OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'frozen vegetables', 'milk'}), confidence=0.0728744939271255, lift=3.0883140053523634), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'milk'}), items_add=frozenset({'olive oil'}), confidence=0.20338983050847456, lift=3.088314005352364), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'olive oil'}), items_add=frozenset({'milk'}), confidence=0.4235294117647058, lift=3.2684095860566447)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'soup', 'milk'}), support=0.003999466737768298, ordered_statistics=[OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'frozen vegetables', 'soup'}), confidence=0.030864197530864203, lift=3.858539094650206), OrderedStatistic(items_base=frozenset({'soup'}), items_add=frozenset({'frozen vegetables', 'milk'}), confidence=0.079155672823219, lift=3.3545011403783374), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'milk'}), items_add=frozenset({'soup'}), confidence=0.16949152542372883, lift=3.354501140378338), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'soup'}), items_add=frozenset({'milk'}), confidence=0.5, lift=3.858539094650206)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'tomatoes', 'milk'}), support=0.0041327822956939075, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'milk', 'tomatoes'}), confidence=0.043356643356643354, lift=3.0973160173160172), OrderedStatistic(items_base=frozenset({'tomatoes', 'milk'}), items_add=frozenset({'frozen vegetables'}), confidence=0.29523809523809524, lift=3.0973160173160172)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'mineral water', 'shrimp'}), support=0.007199040127982935, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'mineral water', 'shrimp'}), confidence=0.07552447552447553, lift=3.2006163328197226), OrderedStatistic(items_base=frozenset({'mineral water', 'shrimp'}), items_add=frozenset({'frozen vegetables'}), confidence=0.30508474576271183, lift=3.200616332819722)]),
     RelationRecord(items=frozenset({'spaghetti', 'frozen vegetables', 'olive oil'}), support=0.005732568990801226, ordered_statistics=[OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'frozen vegetables', 'spaghetti'}), confidence=0.08704453441295547, lift=3.124024175270713), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti'}), items_add=frozenset({'olive oil'}), confidence=0.20574162679425836, lift=3.1240241752707125)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'spaghetti', 'shrimp'}), support=0.005999200106652446, ordered_statistics=[OrderedStatistic(items_base=frozenset({'shrimp'}), items_add=frozenset({'frozen vegetables', 'spaghetti'}), confidence=0.08395522388059701, lift=3.013148968078269), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti'}), items_add=frozenset({'shrimp'}), confidence=0.21531100478468898, lift=3.0131489680782684)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'tomatoes', 'shrimp'}), support=0.003999466737768298, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'shrimp', 'tomatoes'}), confidence=0.04195804195804196, lift=3.7467532467532467), OrderedStatistic(items_base=frozenset({'shrimp'}), items_add=frozenset({'frozen vegetables', 'tomatoes'}), confidence=0.055970149253731345, lift=3.469686690514371), OrderedStatistic(items_base=frozenset({'tomatoes'}), items_add=frozenset({'frozen vegetables', 'shrimp'}), confidence=0.058479532163742694, lift=3.5092397660818713), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'shrimp'}), items_add=frozenset({'tomatoes'}), confidence=0.24000000000000002, lift=3.5092397660818717), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'tomatoes'}), items_add=frozenset({'shrimp'}), confidence=0.2479338842975207, lift=3.4696866905143704), OrderedStatistic(items_base=frozenset({'tomatoes', 'shrimp'}), items_add=frozenset({'frozen vegetables'}), confidence=0.35714285714285715, lift=3.7467532467532467)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'spaghetti', 'tomatoes'}), support=0.006665777896280496, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'spaghetti', 'tomatoes'}), confidence=0.06993006993006994, lift=3.341053850607991), OrderedStatistic(items_base=frozenset({'tomatoes'}), items_add=frozenset({'frozen vegetables', 'spaghetti'}), confidence=0.09746588693957116, lift=3.4980460188216425), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti'}), items_add=frozenset({'tomatoes'}), confidence=0.23923444976076558, lift=3.4980460188216425), OrderedStatistic(items_base=frozenset({'spaghetti', 'tomatoes'}), items_add=frozenset({'frozen vegetables'}), confidence=0.3184713375796179, lift=3.341053850607991)]),
     RelationRecord(items=frozenset({'ground beef', 'grated cheese', 'spaghetti'}), support=0.005332622317024397, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'grated cheese', 'spaghetti'}), confidence=0.054274084124830396, lift=3.283144395325426), OrderedStatistic(items_base=frozenset({'grated cheese', 'spaghetti'}), items_add=frozenset({'ground beef'}), confidence=0.3225806451612903, lift=3.283144395325426)]),
     RelationRecord(items=frozenset({'herb & pepper', 'grated cheese', 'spaghetti'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'grated cheese'}), items_add=frozenset({'herb & pepper', 'spaghetti'}), confidence=0.058524173027989825, lift=3.5982772285487843), OrderedStatistic(items_base=frozenset({'herb & pepper'}), items_add=frozenset({'grated cheese', 'spaghetti'}), confidence=0.06199460916442048, lift=3.7501738979219197), OrderedStatistic(items_base=frozenset({'grated cheese', 'spaghetti'}), items_add=frozenset({'herb & pepper'}), confidence=0.18548387096774194, lift=3.7501738979219197), OrderedStatistic(items_base=frozenset({'herb & pepper', 'spaghetti'}), items_add=frozenset({'grated cheese'}), confidence=0.18852459016393444, lift=3.5982772285487843)]),
     RelationRecord(items=frozenset({'green tea', 'ground beef', 'tomatoes'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'tomatoes'}), items_add=frozenset({'green tea', 'ground beef'}), confidence=0.04483430799220273, lift=3.0297490472929067), OrderedStatistic(items_base=frozenset({'green tea', 'ground beef'}), items_add=frozenset({'tomatoes'}), confidence=0.2072072072072072, lift=3.0297490472929067)]),
     RelationRecord(items=frozenset({'ground beef', 'herb & pepper', 'milk'}), support=0.0035995200639914677, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'herb & pepper', 'milk'}), confidence=0.036635006784260515, lift=3.982596896938234), OrderedStatistic(items_base=frozenset({'herb & pepper'}), items_add=frozenset({'ground beef', 'milk'}), confidence=0.07277628032345013, lift=3.3084538103406027), OrderedStatistic(items_base=frozenset({'ground beef', 'milk'}), items_add=frozenset({'herb & pepper'}), confidence=0.16363636363636364, lift=3.3084538103406027), OrderedStatistic(items_base=frozenset({'herb & pepper', 'milk'}), items_add=frozenset({'ground beef'}), confidence=0.3913043478260869, lift=3.9825968969382335)]),
     RelationRecord(items=frozenset({'ground beef', 'herb & pepper', 'mineral water'}), support=0.006665777896280496, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'herb & pepper', 'mineral water'}), confidence=0.067842605156038, lift=3.9756826662143836), OrderedStatistic(items_base=frozenset({'herb & pepper'}), items_add=frozenset({'ground beef', 'mineral water'}), confidence=0.1347708894878706, lift=3.292887433382793), OrderedStatistic(items_base=frozenset({'ground beef', 'mineral water'}), items_add=frozenset({'herb & pepper'}), confidence=0.16286644951140067, lift=3.2928874333827935), OrderedStatistic(items_base=frozenset({'herb & pepper', 'mineral water'}), items_add=frozenset({'ground beef'}), confidence=0.39062500000000006, lift=3.975682666214383)]),
     RelationRecord(items=frozenset({'ground beef', 'herb & pepper', 'spaghetti'}), support=0.006399146780429276, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'herb & pepper', 'spaghetti'}), confidence=0.06512890094979648, lift=4.004359721511667), OrderedStatistic(items_base=frozenset({'herb & pepper'}), items_add=frozenset({'ground beef', 'spaghetti'}), confidence=0.1293800539083558, lift=3.3009516475053635), OrderedStatistic(items_base=frozenset({'ground beef', 'spaghetti'}), items_add=frozenset({'herb & pepper'}), confidence=0.163265306122449, lift=3.3009516475053635), OrderedStatistic(items_base=frozenset({'herb & pepper', 'spaghetti'}), items_add=frozenset({'ground beef'}), confidence=0.3934426229508197, lift=4.004359721511667)]),
     RelationRecord(items=frozenset({'ground beef', 'olive oil', 'milk'}), support=0.004932675643247567, ordered_statistics=[OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'ground beef', 'milk'}), confidence=0.07489878542510121, lift=3.4049441786283894), OrderedStatistic(items_base=frozenset({'ground beef', 'milk'}), items_add=frozenset({'olive oil'}), confidence=0.22424242424242427, lift=3.40494417862839)]),
     RelationRecord(items=frozenset({'ground beef', 'soup', 'milk'}), support=0.003999466737768298, ordered_statistics=[OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'ground beef', 'soup'}), confidence=0.030864197530864203, lift=3.1714019956029094), OrderedStatistic(items_base=frozenset({'soup'}), items_add=frozenset({'ground beef', 'milk'}), confidence=0.079155672823219, lift=3.5984648596785807), OrderedStatistic(items_base=frozenset({'ground beef', 'milk'}), items_add=frozenset({'soup'}), confidence=0.18181818181818185, lift=3.5984648596785807), OrderedStatistic(items_base=frozenset({'ground beef', 'soup'}), items_add=frozenset({'milk'}), confidence=0.4109589041095891, lift=3.1714019956029094)]),
     RelationRecord(items=frozenset({'ground beef', 'pepper', 'spaghetti'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'pepper', 'spaghetti'}), confidence=0.033921302578019, lift=3.438428251861088), OrderedStatistic(items_base=frozenset({'pepper'}), items_add=frozenset({'ground beef', 'spaghetti'}), confidence=0.12562814070351758, lift=3.2052268143438276), OrderedStatistic(items_base=frozenset({'ground beef', 'spaghetti'}), items_add=frozenset({'pepper'}), confidence=0.08503401360544217, lift=3.2052268143438276), OrderedStatistic(items_base=frozenset({'pepper', 'spaghetti'}), items_add=frozenset({'ground beef'}), confidence=0.33783783783783783, lift=3.4384282518610876)]),
     RelationRecord(items=frozenset({'ground beef', 'spaghetti', 'shrimp'}), support=0.005999200106652446, ordered_statistics=[OrderedStatistic(items_base=frozenset({'spaghetti'}), items_add=frozenset({'ground beef', 'shrimp'}), confidence=0.03445635528330781, lift=3.0053153602336264), OrderedStatistic(items_base=frozenset({'ground beef', 'shrimp'}), items_add=frozenset({'spaghetti'}), confidence=0.5232558139534884, lift=3.005315360233627)]),
     RelationRecord(items=frozenset({'ground beef', 'spaghetti', 'tomato sauce'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'spaghetti', 'tomato sauce'}), confidence=0.03120759837177748, lift=4.980599901844742), OrderedStatistic(items_base=frozenset({'spaghetti'}), items_add=frozenset({'ground beef', 'tomato sauce'}), confidence=0.01761102603369066, lift=3.302507656967841), OrderedStatistic(items_base=frozenset({'tomato sauce'}), items_add=frozenset({'ground beef', 'spaghetti'}), confidence=0.2169811320754717, lift=5.535970992170453), OrderedStatistic(items_base=frozenset({'ground beef', 'spaghetti'}), items_add=frozenset({'tomato sauce'}), confidence=0.0782312925170068, lift=5.535970992170453), OrderedStatistic(items_base=frozenset({'ground beef', 'tomato sauce'}), items_add=frozenset({'spaghetti'}), confidence=0.5750000000000001, lift=3.3025076569678413), OrderedStatistic(items_base=frozenset({'spaghetti', 'tomato sauce'}), items_add=frozenset({'ground beef'}), confidence=0.4893617021276596, lift=4.980599901844742)]),
     RelationRecord(items=frozenset({'light cream', 'mineral water', 'spaghetti'}), support=0.003199573390214638, ordered_statistics=[OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.20512820512820515, lift=3.4345238095238098), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'light cream'}), confidence=0.053571428571428575, lift=3.4345238095238098)]),
     RelationRecord(items=frozenset({'mineral water', 'soup', 'milk'}), support=0.008532195707239034, ordered_statistics=[OrderedStatistic(items_base=frozenset({'soup'}), items_add=frozenset({'mineral water', 'milk'}), confidence=0.16886543535620052, lift=3.518498973907945), OrderedStatistic(items_base=frozenset({'mineral water', 'milk'}), items_add=frozenset({'soup'}), confidence=0.17777777777777776, lift=3.5184989739079446)]),
     RelationRecord(items=frozenset({'olive oil', 'shrimp', 'milk'}), support=0.003199573390214638, ordered_statistics=[OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'olive oil', 'shrimp'}), confidence=0.02469135802469136, lift=3.0362274843149164), OrderedStatistic(items_base=frozenset({'olive oil', 'shrimp'}), items_add=frozenset({'milk'}), confidence=0.3934426229508197, lift=3.0362274843149164)]),
     RelationRecord(items=frozenset({'soup', 'olive oil', 'milk'}), support=0.0035995200639914677, ordered_statistics=[OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'soup', 'olive oil'}), confidence=0.02777777777777778, lift=3.1098673300165838), OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'soup', 'milk'}), confidence=0.054655870445344125, lift=3.596260387811634), OrderedStatistic(items_base=frozenset({'soup'}), items_add=frozenset({'olive oil', 'milk'}), confidence=0.07124010554089709, lift=4.174781497361478), OrderedStatistic(items_base=frozenset({'olive oil', 'milk'}), items_add=frozenset({'soup'}), confidence=0.2109375, lift=4.174781497361478), OrderedStatistic(items_base=frozenset({'soup', 'milk'}), items_add=frozenset({'olive oil'}), confidence=0.23684210526315788, lift=3.5962603878116344), OrderedStatistic(items_base=frozenset({'soup', 'olive oil'}), items_add=frozenset({'milk'}), confidence=0.4029850746268656, lift=3.1098673300165833)]),
     RelationRecord(items=frozenset({'spaghetti', 'olive oil', 'milk'}), support=0.007199040127982935, ordered_statistics=[OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'spaghetti', 'milk'}), confidence=0.10931174089068825, lift=3.0825089038385434), OrderedStatistic(items_base=frozenset({'spaghetti', 'milk'}), items_add=frozenset({'olive oil'}), confidence=0.20300751879699247, lift=3.0825089038385434)]),
     RelationRecord(items=frozenset({'soup', 'tomatoes', 'milk'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'soup', 'tomatoes'}), confidence=0.02366255144032922, lift=3.4133230452674903), OrderedStatistic(items_base=frozenset({'soup'}), items_add=frozenset({'milk', 'tomatoes'}), confidence=0.06068601583113457, lift=4.335293378565146), OrderedStatistic(items_base=frozenset({'tomatoes', 'milk'}), items_add=frozenset({'soup'}), confidence=0.21904761904761905, lift=4.335293378565146), OrderedStatistic(items_base=frozenset({'soup', 'tomatoes'}), items_add=frozenset({'milk'}), confidence=0.44230769230769235, lift=3.4133230452674903)]),
     RelationRecord(items=frozenset({'whole wheat pasta', 'spaghetti', 'milk'}), support=0.003999466737768298, ordered_statistics=[OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'whole wheat pasta', 'spaghetti'}), confidence=0.030864197530864203, lift=3.5077628133183696), OrderedStatistic(items_base=frozenset({'whole wheat pasta'}), items_add=frozenset({'spaghetti', 'milk'}), confidence=0.13574660633484165, lift=3.8279522335249894), OrderedStatistic(items_base=frozenset({'spaghetti', 'milk'}), items_add=frozenset({'whole wheat pasta'}), confidence=0.11278195488721805, lift=3.827952233524989), OrderedStatistic(items_base=frozenset({'whole wheat pasta', 'spaghetti'}), items_add=frozenset({'milk'}), confidence=0.4545454545454546, lift=3.5077628133183696)]),
     RelationRecord(items=frozenset({'mineral water', 'soup', 'olive oil'}), support=0.005199306759098787, ordered_statistics=[OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'mineral water', 'soup'}), confidence=0.07894736842105264, lift=3.423030118649225), OrderedStatistic(items_base=frozenset({'soup'}), items_add=frozenset({'mineral water', 'olive oil'}), confidence=0.1029023746701847, lift=3.7288440212611373), OrderedStatistic(items_base=frozenset({'mineral water', 'olive oil'}), items_add=frozenset({'soup'}), confidence=0.18840579710144928, lift=3.7288440212611373), OrderedStatistic(items_base=frozenset({'mineral water', 'soup'}), items_add=frozenset({'olive oil'}), confidence=0.22543352601156072, lift=3.4230301186492245)]),
     RelationRecord(items=frozenset({'whole wheat pasta', 'mineral water', 'olive oil'}), support=0.0038661511798426876, ordered_statistics=[OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'whole wheat pasta', 'mineral water'}), confidence=0.058704453441295545, lift=6.115862573099415), OrderedStatistic(items_base=frozenset({'whole wheat pasta'}), items_add=frozenset({'mineral water', 'olive oil'}), confidence=0.13122171945701358, lift=4.755044046604149), OrderedStatistic(items_base=frozenset({'mineral water', 'olive oil'}), items_add=frozenset({'whole wheat pasta'}), confidence=0.1400966183574879, lift=4.755044046604148), OrderedStatistic(items_base=frozenset({'whole wheat pasta', 'mineral water'}), items_add=frozenset({'olive oil'}), confidence=0.4027777777777778, lift=6.115862573099416)]),
     RelationRecord(items=frozenset({'mineral water', 'soup', 'tomatoes'}), support=0.0037328356219170776, ordered_statistics=[OrderedStatistic(items_base=frozenset({'soup'}), items_add=frozenset({'mineral water', 'tomatoes'}), confidence=0.07387862796833773, lift=3.0282163300027394), OrderedStatistic(items_base=frozenset({'mineral water', 'tomatoes'}), items_add=frozenset({'soup'}), confidence=0.15300546448087432, lift=3.0282163300027394)]),
     RelationRecord(items=frozenset({'spaghetti', 'olive oil', 'pancakes'}), support=0.005065991201173177, ordered_statistics=[OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'spaghetti', 'pancakes'}), confidence=0.07692307692307693, lift=3.052910052910053), OrderedStatistic(items_base=frozenset({'spaghetti', 'pancakes'}), items_add=frozenset({'olive oil'}), confidence=0.20105820105820105, lift=3.0529100529100526)]),
     RelationRecord(items=frozenset({'spaghetti', 'olive oil', 'tomatoes'}), support=0.004399413411545127, ordered_statistics=[OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'spaghetti', 'tomatoes'}), confidence=0.06680161943319839, lift=3.1915856520281602), OrderedStatistic(items_base=frozenset({'spaghetti'}), items_add=frozenset({'olive oil', 'tomatoes'}), confidence=0.02526799387442573, lift=3.5099115194827295), OrderedStatistic(items_base=frozenset({'olive oil', 'tomatoes'}), items_add=frozenset({'spaghetti'}), confidence=0.6111111111111112, lift=3.5099115194827295), OrderedStatistic(items_base=frozenset({'spaghetti', 'tomatoes'}), items_add=frozenset({'olive oil'}), confidence=0.21019108280254778, lift=3.19158565202816)]),
     RelationRecord(items=frozenset({'whole wheat rice', 'spaghetti', 'tomatoes'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'tomatoes'}), items_add=frozenset({'whole wheat rice', 'spaghetti'}), confidence=0.04483430799220273, lift=3.1726617382029496), OrderedStatistic(items_base=frozenset({'whole wheat rice', 'spaghetti'}), items_add=frozenset({'tomatoes'}), confidence=0.2169811320754717, lift=3.1726617382029496)]),
     RelationRecord(items=frozenset({'eggs', 'ground beef', 'mineral water', 'chocolate'}), support=0.003999466737768298, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'eggs', 'mineral water', 'chocolate'}), confidence=0.0407055630936228, lift=3.023093354111531), OrderedStatistic(items_base=frozenset({'ground beef', 'chocolate'}), items_add=frozenset({'eggs', 'mineral water'}), confidence=0.17341040462427748, lift=3.4051084949913752), OrderedStatistic(items_base=frozenset({'mineral water', 'chocolate'}), items_add=frozenset({'eggs', 'ground beef'}), confidence=0.0759493670886076, lift=3.7979746835443042), OrderedStatistic(items_base=frozenset({'eggs', 'ground beef'}), items_add=frozenset({'mineral water', 'chocolate'}), confidence=0.20000000000000004, lift=3.7979746835443047), OrderedStatistic(items_base=frozenset({'eggs', 'mineral water'}), items_add=frozenset({'ground beef', 'chocolate'}), confidence=0.07853403141361257, lift=3.405108494991375), OrderedStatistic(items_base=frozenset({'eggs', 'mineral water', 'chocolate'}), items_add=frozenset({'ground beef'}), confidence=0.29702970297029707, lift=3.023093354111531)]),
     RelationRecord(items=frozenset({'eggs', 'ground beef', 'spaghetti', 'chocolate'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef', 'chocolate'}), items_add=frozenset({'eggs', 'spaghetti'}), confidence=0.13294797687861273, lift=3.6395721699506356), OrderedStatistic(items_base=frozenset({'spaghetti', 'chocolate'}), items_add=frozenset({'eggs', 'ground beef'}), confidence=0.0782312925170068, lift=3.912086167800454), OrderedStatistic(items_base=frozenset({'eggs', 'ground beef'}), items_add=frozenset({'spaghetti', 'chocolate'}), confidence=0.15333333333333335, lift=3.912086167800454), OrderedStatistic(items_base=frozenset({'eggs', 'spaghetti'}), items_add=frozenset({'ground beef', 'chocolate'}), confidence=0.08394160583941607, lift=3.6395721699506356)]),
     RelationRecord(items=frozenset({'ground beef', 'frozen vegetables', 'mineral water', 'chocolate'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'ground beef', 'mineral water', 'chocolate'}), confidence=0.03496503496503497, lift=3.1984478935698455), OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'frozen vegetables', 'mineral water', 'chocolate'}), confidence=0.033921302578019, lift=3.4855300087358976), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'chocolate'}), items_add=frozenset({'ground beef', 'mineral water'}), confidence=0.14534883720930233, lift=3.5513408075145825), OrderedStatistic(items_base=frozenset({'ground beef', 'chocolate'}), items_add=frozenset({'frozen vegetables', 'mineral water'}), confidence=0.14450867052023122, lift=4.044625140194979), OrderedStatistic(items_base=frozenset({'mineral water', 'chocolate'}), items_add=frozenset({'ground beef', 'frozen vegetables'}), confidence=0.06329113924050633, lift=3.738164058606598), OrderedStatistic(items_base=frozenset({'ground beef', 'frozen vegetables'}), items_add=frozenset({'mineral water', 'chocolate'}), confidence=0.19685039370078738, lift=3.738164058606598), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water'}), items_add=frozenset({'ground beef', 'chocolate'}), confidence=0.09328358208955223, lift=4.044625140194979), OrderedStatistic(items_base=frozenset({'ground beef', 'mineral water'}), items_add=frozenset({'frozen vegetables', 'chocolate'}), confidence=0.08143322475570033, lift=3.551340807514583), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water', 'chocolate'}), items_add=frozenset({'ground beef'}), confidence=0.34246575342465757, lift=3.4855300087358976), OrderedStatistic(items_base=frozenset({'ground beef', 'mineral water', 'chocolate'}), items_add=frozenset({'frozen vegetables'}), confidence=0.30487804878048785, lift=3.1984478935698455)]),
     RelationRecord(items=frozenset({'ground beef', 'frozen vegetables', 'spaghetti', 'chocolate'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'ground beef', 'spaghetti', 'chocolate'}), confidence=0.03216783216783217, lift=3.4969696969696975), OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'frozen vegetables', 'spaghetti', 'chocolate'}), confidence=0.03120759837177748, lift=3.967596531978015), OrderedStatistic(items_base=frozenset({'spaghetti'}), items_add=frozenset({'ground beef', 'frozen vegetables', 'chocolate'}), confidence=0.01761102603369066, lift=3.0721001460165964), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'chocolate'}), items_add=frozenset({'ground beef', 'spaghetti'}), confidence=0.13372093023255816, lift=3.411703053314349), OrderedStatistic(items_base=frozenset({'ground beef', 'chocolate'}), items_add=frozenset({'frozen vegetables', 'spaghetti'}), confidence=0.13294797687861273, lift=4.771496529026192), OrderedStatistic(items_base=frozenset({'spaghetti', 'chocolate'}), items_add=frozenset({'ground beef', 'frozen vegetables'}), confidence=0.0782312925170068, lift=4.620574213937544), OrderedStatistic(items_base=frozenset({'ground beef', 'frozen vegetables'}), items_add=frozenset({'spaghetti', 'chocolate'}), confidence=0.1811023622047244, lift=4.620574213937543), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti'}), items_add=frozenset({'ground beef', 'chocolate'}), confidence=0.11004784688995216, lift=4.771496529026192), OrderedStatistic(items_base=frozenset({'ground beef', 'spaghetti'}), items_add=frozenset({'frozen vegetables', 'chocolate'}), confidence=0.0782312925170068, lift=3.4117030533143495), OrderedStatistic(items_base=frozenset({'ground beef', 'frozen vegetables', 'chocolate'}), items_add=frozenset({'spaghetti'}), confidence=0.5348837209302326, lift=3.0721001460165964), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti', 'chocolate'}), items_add=frozenset({'ground beef'}), confidence=0.3898305084745763, lift=3.967596531978015), OrderedStatistic(items_base=frozenset({'ground beef', 'spaghetti', 'chocolate'}), items_add=frozenset({'frozen vegetables'}), confidence=0.33333333333333337, lift=3.4969696969696975)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'mineral water', 'chocolate', 'milk'}), support=0.003999466737768298, ordered_statistics=[OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'frozen vegetables', 'mineral water', 'chocolate'}), confidence=0.030864197530864203, lift=3.1714019956029094), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'chocolate'}), items_add=frozenset({'mineral water', 'milk'}), confidence=0.17441860465116282, lift=3.63420542635659), OrderedStatistic(items_base=frozenset({'chocolate', 'milk'}), items_add=frozenset({'frozen vegetables', 'mineral water'}), confidence=0.1244813278008299, lift=3.4840837307239743), OrderedStatistic(items_base=frozenset({'mineral water', 'chocolate'}), items_add=frozenset({'frozen vegetables', 'milk'}), confidence=0.0759493670886076, lift=3.218622613173139), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'milk'}), items_add=frozenset({'mineral water', 'chocolate'}), confidence=0.16949152542372883, lift=3.2186226131731392), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water'}), items_add=frozenset({'chocolate', 'milk'}), confidence=0.11194029850746269, lift=3.484083730723974), OrderedStatistic(items_base=frozenset({'mineral water', 'milk'}), items_add=frozenset({'frozen vegetables', 'chocolate'}), confidence=0.08333333333333334, lift=3.6342054263565897), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water', 'chocolate'}), items_add=frozenset({'milk'}), confidence=0.4109589041095891, lift=3.1714019956029094)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'spaghetti', 'chocolate', 'milk'}), support=0.0034662045060658577, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'spaghetti', 'chocolate', 'milk'}), confidence=0.03636363636363636, lift=3.326385809312639), OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'frozen vegetables', 'spaghetti', 'chocolate'}), confidence=0.026748971193415638, lift=3.400746320708656), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'chocolate'}), items_add=frozenset({'spaghetti', 'milk'}), confidence=0.1511627906976744, lift=4.262677041440811), OrderedStatistic(items_base=frozenset({'chocolate', 'milk'}), items_add=frozenset({'frozen vegetables', 'spaghetti'}), confidence=0.1078838174273859, lift=3.87194504556374), OrderedStatistic(items_base=frozenset({'spaghetti', 'chocolate'}), items_add=frozenset({'frozen vegetables', 'milk'}), confidence=0.08843537414965985, lift=3.747761251393212), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'milk'}), items_add=frozenset({'spaghetti', 'chocolate'}), confidence=0.14689265536723162, lift=3.747761251393212), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti'}), items_add=frozenset({'chocolate', 'milk'}), confidence=0.12440191387559808, lift=3.87194504556374), OrderedStatistic(items_base=frozenset({'spaghetti', 'milk'}), items_add=frozenset({'frozen vegetables', 'chocolate'}), confidence=0.09774436090225563, lift=4.262677041440812), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti', 'chocolate'}), items_add=frozenset({'milk'}), confidence=0.44067796610169485, lift=3.4007463207086555), OrderedStatistic(items_base=frozenset({'spaghetti', 'chocolate', 'milk'}), items_add=frozenset({'frozen vegetables'}), confidence=0.3170731707317073, lift=3.3263858093126384)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'mineral water', 'chocolate', 'shrimp'}), support=0.003199573390214638, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'mineral water', 'chocolate', 'shrimp'}), confidence=0.033566433566433566, lift=4.417224880382775), OrderedStatistic(items_base=frozenset({'shrimp'}), items_add=frozenset({'frozen vegetables', 'mineral water', 'chocolate'}), confidence=0.04477611940298507, lift=4.600899611531384), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'chocolate'}), items_add=frozenset({'mineral water', 'shrimp'}), confidence=0.13953488372093026, lift=5.913283405597163), OrderedStatistic(items_base=frozenset({'mineral water', 'chocolate'}), items_add=frozenset({'frozen vegetables', 'shrimp'}), confidence=0.06075949367088608, lift=3.6460556962025317), OrderedStatistic(items_base=frozenset({'chocolate', 'shrimp'}), items_add=frozenset({'frozen vegetables', 'mineral water'}), confidence=0.1777777777777778, lift=4.975787728026535), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water'}), items_add=frozenset({'chocolate', 'shrimp'}), confidence=0.08955223880597014, lift=4.975787728026535), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'shrimp'}), items_add=frozenset({'mineral water', 'chocolate'}), confidence=0.192, lift=3.6460556962025317), OrderedStatistic(items_base=frozenset({'mineral water', 'shrimp'}), items_add=frozenset({'frozen vegetables', 'chocolate'}), confidence=0.13559322033898305, lift=5.913283405597162), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water', 'chocolate'}), items_add=frozenset({'shrimp'}), confidence=0.32876712328767127, lift=4.600899611531385), OrderedStatistic(items_base=frozenset({'mineral water', 'chocolate', 'shrimp'}), items_add=frozenset({'frozen vegetables'}), confidence=0.4210526315789474, lift=4.417224880382776)]),
     RelationRecord(items=frozenset({'spaghetti', 'frozen vegetables', 'mineral water', 'chocolate'}), support=0.0041327822956939075, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables', 'chocolate'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.1802325581395349, lift=3.0176884343853825), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'frozen vegetables', 'chocolate'}), confidence=0.06919642857142858, lift=3.0176884343853825)]),
     RelationRecord(items=frozenset({'ground beef', 'mineral water', 'chocolate', 'milk'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef', 'chocolate'}), items_add=frozenset({'mineral water', 'milk'}), confidence=0.14450867052023122, lift=3.0109987154784847), OrderedStatistic(items_base=frozenset({'mineral water', 'milk'}), items_add=frozenset({'ground beef', 'chocolate'}), confidence=0.06944444444444445, lift=3.0109987154784843)]),
     RelationRecord(items=frozenset({'spaghetti', 'mineral water', 'olive oil', 'chocolate'}), support=0.0038661511798426876, ordered_statistics=[OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'mineral water', 'spaghetti', 'chocolate'}), confidence=0.058704453441295545, lift=3.700353825740822), OrderedStatistic(items_base=frozenset({'mineral water', 'chocolate'}), items_add=frozenset({'olive oil', 'spaghetti'}), confidence=0.07341772151898734, lift=3.201780983220489), OrderedStatistic(items_base=frozenset({'olive oil', 'chocolate'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.23577235772357724, lift=3.947608159117306), OrderedStatistic(items_base=frozenset({'spaghetti', 'chocolate'}), items_add=frozenset({'mineral water', 'olive oil'}), confidence=0.09863945578231292, lift=3.5743698445561796), OrderedStatistic(items_base=frozenset({'mineral water', 'olive oil'}), items_add=frozenset({'spaghetti', 'chocolate'}), confidence=0.1400966183574879, lift=3.5743698445561796), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'olive oil', 'chocolate'}), confidence=0.06473214285714286, lift=3.947608159117306), OrderedStatistic(items_base=frozenset({'spaghetti', 'olive oil'}), items_add=frozenset({'mineral water', 'chocolate'}), confidence=0.1686046511627907, lift=3.2017809832204884), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti', 'chocolate'}), items_add=frozenset({'olive oil'}), confidence=0.2436974789915966, lift=3.700353825740822)]),
     RelationRecord(items=frozenset({'mineral water', 'spaghetti', 'chocolate', 'pancakes'}), support=0.0037328356219170776, ordered_statistics=[OrderedStatistic(items_base=frozenset({'chocolate', 'pancakes'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.1879194630872483, lift=3.1463926174496644), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'chocolate', 'pancakes'}), confidence=0.0625, lift=3.1463926174496644)]),
     RelationRecord(items=frozenset({'mineral water', 'spaghetti', 'chocolate', 'shrimp'}), support=0.0034662045060658577, ordered_statistics=[OrderedStatistic(items_base=frozenset({'shrimp'}), items_add=frozenset({'mineral water', 'spaghetti', 'chocolate'}), confidence=0.04850746268656716, lift=3.0576006522011783), OrderedStatistic(items_base=frozenset({'mineral water', 'chocolate'}), items_add=frozenset({'spaghetti', 'shrimp'}), confidence=0.06582278481012657, lift=3.10526231987899), OrderedStatistic(items_base=frozenset({'chocolate', 'shrimp'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.1925925925925926, lift=3.2246362433862434), OrderedStatistic(items_base=frozenset({'spaghetti', 'chocolate'}), items_add=frozenset({'mineral water', 'shrimp'}), confidence=0.08843537414965985, lift=3.747761251393212), OrderedStatistic(items_base=frozenset({'mineral water', 'shrimp'}), items_add=frozenset({'spaghetti', 'chocolate'}), confidence=0.14689265536723162, lift=3.747761251393212), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'chocolate', 'shrimp'}), confidence=0.05803571428571428, lift=3.2246362433862434), OrderedStatistic(items_base=frozenset({'spaghetti', 'shrimp'}), items_add=frozenset({'mineral water', 'chocolate'}), confidence=0.16352201257861634, lift=3.10526231987899), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti', 'chocolate'}), items_add=frozenset({'shrimp'}), confidence=0.21848739495798317, lift=3.0576006522011783)]),
     RelationRecord(items=frozenset({'eggs', 'frozen vegetables', 'mineral water', 'milk'}), support=0.0037328356219170776, ordered_statistics=[OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'eggs', 'frozen vegetables', 'mineral water'}), confidence=0.02880658436213992, lift=3.177620430888405), OrderedStatistic(items_base=frozenset({'eggs', 'frozen vegetables'}), items_add=frozenset({'mineral water', 'milk'}), confidence=0.17177914110429446, lift=3.5792092706203134), OrderedStatistic(items_base=frozenset({'eggs', 'milk'}), items_add=frozenset({'frozen vegetables', 'mineral water'}), confidence=0.12121212121212122, lift=3.392582541836273), OrderedStatistic(items_base=frozenset({'eggs', 'mineral water'}), items_add=frozenset({'frozen vegetables', 'milk'}), confidence=0.07329842931937172, lift=3.106279764545804), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'milk'}), items_add=frozenset({'eggs', 'mineral water'}), confidence=0.1581920903954802, lift=3.106279764545804), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water'}), items_add=frozenset({'eggs', 'milk'}), confidence=0.10447761194029849, lift=3.3925825418362727), OrderedStatistic(items_base=frozenset({'mineral water', 'milk'}), items_add=frozenset({'eggs', 'frozen vegetables'}), confidence=0.07777777777777778, lift=3.5792092706203134), OrderedStatistic(items_base=frozenset({'eggs', 'frozen vegetables', 'mineral water'}), items_add=frozenset({'milk'}), confidence=0.411764705882353, lift=3.177620430888405)]),
     RelationRecord(items=frozenset({'eggs', 'ground beef', 'mineral water', 'milk'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'eggs', 'ground beef'}), items_add=frozenset({'mineral water', 'milk'}), confidence=0.16666666666666669, lift=3.4726851851851857), OrderedStatistic(items_base=frozenset({'mineral water', 'milk'}), items_add=frozenset({'eggs', 'ground beef'}), confidence=0.06944444444444445, lift=3.4726851851851857)]),
     RelationRecord(items=frozenset({'eggs', 'ground beef', 'mineral water', 'spaghetti'}), support=0.0038661511798426876, ordered_statistics=[OrderedStatistic(items_base=frozenset({'eggs', 'ground beef'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.19333333333333336, lift=3.237038690476191), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'eggs', 'ground beef'}), confidence=0.06473214285714286, lift=3.237038690476191)]),
     RelationRecord(items=frozenset({'frozen smoothie', 'mineral water', 'spaghetti', 'milk'}), support=0.003199573390214638, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen smoothie'}), items_add=frozenset({'mineral water', 'spaghetti', 'milk'}), confidence=0.05052631578947369, lift=3.2118465655664585), OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'frozen smoothie', 'mineral water', 'spaghetti'}), confidence=0.02469135802469136, lift=3.6315662067296057), OrderedStatistic(items_base=frozenset({'frozen smoothie', 'milk'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.22429906542056074, lift=3.7555073431241657), OrderedStatistic(items_base=frozenset({'frozen smoothie', 'mineral water'}), items_add=frozenset({'spaghetti', 'milk'}), confidence=0.15789473684210528, lift=4.452512861100119), OrderedStatistic(items_base=frozenset({'frozen smoothie', 'spaghetti'}), items_add=frozenset({'mineral water', 'milk'}), confidence=0.20512820512820515, lift=4.274074074074075), OrderedStatistic(items_base=frozenset({'mineral water', 'milk'}), items_add=frozenset({'frozen smoothie', 'spaghetti'}), confidence=0.06666666666666668, lift=4.274074074074075), OrderedStatistic(items_base=frozenset({'spaghetti', 'milk'}), items_add=frozenset({'frozen smoothie', 'mineral water'}), confidence=0.09022556390977443, lift=4.452512861100119), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'frozen smoothie', 'milk'}), confidence=0.053571428571428575, lift=3.7555073431241657), OrderedStatistic(items_base=frozenset({'frozen smoothie', 'mineral water', 'spaghetti'}), items_add=frozenset({'milk'}), confidence=0.4705882352941177, lift=3.631566206729606), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti', 'milk'}), items_add=frozenset({'frozen smoothie'}), confidence=0.2033898305084746, lift=3.211846565566459)]),
     RelationRecord(items=frozenset({'ground beef', 'frozen vegetables', 'mineral water', 'milk'}), support=0.0037328356219170776, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'ground beef', 'mineral water', 'milk'}), confidence=0.03916083916083916, lift=3.5391018619934282), OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'frozen vegetables', 'mineral water', 'milk'}), confidence=0.037991858887381276, lift=3.433457030292132), OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'ground beef', 'frozen vegetables', 'mineral water'}), confidence=0.02880658436213992, lift=3.1315679608755294), OrderedStatistic(items_base=frozenset({'ground beef', 'frozen vegetables'}), items_add=frozenset({'mineral water', 'milk'}), confidence=0.22047244094488186, lift=4.593788276465442), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'milk'}), items_add=frozenset({'ground beef', 'mineral water'}), confidence=0.1581920903954802, lift=3.865142899206831), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water'}), items_add=frozenset({'ground beef', 'milk'}), confidence=0.10447761194029849, lift=4.749615558570782), OrderedStatistic(items_base=frozenset({'ground beef', 'milk'}), items_add=frozenset({'frozen vegetables', 'mineral water'}), confidence=0.1696969696969697, lift=4.749615558570782), OrderedStatistic(items_base=frozenset({'ground beef', 'mineral water'}), items_add=frozenset({'frozen vegetables', 'milk'}), confidence=0.09120521172638436, lift=3.865142899206831), OrderedStatistic(items_base=frozenset({'mineral water', 'milk'}), items_add=frozenset({'ground beef', 'frozen vegetables'}), confidence=0.07777777777777778, lift=4.593788276465442), OrderedStatistic(items_base=frozenset({'ground beef', 'frozen vegetables', 'mineral water'}), items_add=frozenset({'milk'}), confidence=0.40579710144927533, lift=3.1315679608755294), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water', 'milk'}), items_add=frozenset({'ground beef'}), confidence=0.3373493975903614, lift=3.4334570302921317), OrderedStatistic(items_base=frozenset({'ground beef', 'mineral water', 'milk'}), items_add=frozenset({'frozen vegetables'}), confidence=0.3373493975903614, lift=3.539101861993428)]),
     RelationRecord(items=frozenset({'ground beef', 'frozen vegetables', 'spaghetti', 'milk'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'ground beef', 'spaghetti', 'milk'}), confidence=0.03216783216783217, lift=3.3053549190535496), OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'frozen vegetables', 'spaghetti', 'milk'}), confidence=0.03120759837177748, lift=3.77561605462424), OrderedStatistic(items_base=frozenset({'spaghetti'}), items_add=frozenset({'ground beef', 'frozen vegetables', 'milk'}), confidence=0.01761102603369066, lift=3.0721001460165964), OrderedStatistic(items_base=frozenset({'ground beef', 'frozen vegetables'}), items_add=frozenset({'spaghetti', 'milk'}), confidence=0.1811023622047244, lift=5.1069504469836), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'milk'}), items_add=frozenset({'ground beef', 'spaghetti'}), confidence=0.12994350282485875, lift=3.315327260847842), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti'}), items_add=frozenset({'ground beef', 'milk'}), confidence=0.11004784688995216, lift=5.002841815282007), OrderedStatistic(items_base=frozenset({'ground beef', 'milk'}), items_add=frozenset({'frozen vegetables', 'spaghetti'}), confidence=0.1393939393939394, lift=5.002841815282007), OrderedStatistic(items_base=frozenset({'ground beef', 'spaghetti'}), items_add=frozenset({'frozen vegetables', 'milk'}), confidence=0.0782312925170068, lift=3.315327260847842), OrderedStatistic(items_base=frozenset({'spaghetti', 'milk'}), items_add=frozenset({'ground beef', 'frozen vegetables'}), confidence=0.08646616541353383, lift=5.1069504469836), OrderedStatistic(items_base=frozenset({'ground beef', 'frozen vegetables', 'milk'}), items_add=frozenset({'spaghetti'}), confidence=0.5348837209302326, lift=3.0721001460165964), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti', 'milk'}), items_add=frozenset({'ground beef'}), confidence=0.3709677419354839, lift=3.7756160546242397), OrderedStatistic(items_base=frozenset({'ground beef', 'spaghetti', 'milk'}), items_add=frozenset({'frozen vegetables'}), confidence=0.3150684931506849, lift=3.305354919053549)]),
     RelationRecord(items=frozenset({'ground beef', 'frozen vegetables', 'spaghetti', 'mineral water'}), support=0.004399413411545127, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'frozen vegetables', 'spaghetti', 'mineral water'}), confidence=0.04477611940298508, lift=3.731840796019901), OrderedStatistic(items_base=frozenset({'ground beef', 'frozen vegetables'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.25984251968503935, lift=4.350622187851519), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water'}), items_add=frozenset({'ground beef', 'spaghetti'}), confidence=0.12313432835820895, lift=3.1416006701187937), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti'}), items_add=frozenset({'ground beef', 'mineral water'}), confidence=0.15789473684210528, lift=3.857877593005315), OrderedStatistic(items_base=frozenset({'ground beef', 'mineral water'}), items_add=frozenset({'frozen vegetables', 'spaghetti'}), confidence=0.10749185667752444, lift=3.857877593005315), OrderedStatistic(items_base=frozenset({'ground beef', 'spaghetti'}), items_add=frozenset({'frozen vegetables', 'mineral water'}), confidence=0.11224489795918367, lift=3.1416006701187937), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'ground beef', 'frozen vegetables'}), confidence=0.07366071428571429, lift=4.350622187851518), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti', 'mineral water'}), items_add=frozenset({'ground beef'}), confidence=0.3666666666666667, lift=3.7318407960199007)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'olive oil', 'mineral water', 'milk'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'mineral water', 'olive oil', 'milk'}), confidence=0.03496503496503497, lift=4.098011363636364), OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'frozen vegetables', 'olive oil', 'mineral water'}), confidence=0.025720164609053502, lift=3.9372847904593944), OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'frozen vegetables', 'mineral water', 'milk'}), confidence=0.05060728744939272, lift=4.573557387444516), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'milk'}), items_add=frozenset({'mineral water', 'olive oil'}), confidence=0.14124293785310735, lift=5.118180081334097), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water'}), items_add=frozenset({'olive oil', 'milk'}), confidence=0.09328358208955223, lift=5.4665636660447765), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'olive oil'}), items_add=frozenset({'mineral water', 'milk'}), confidence=0.29411764705882354, lift=6.12826797385621), OrderedStatistic(items_base=frozenset({'mineral water', 'milk'}), items_add=frozenset({'frozen vegetables', 'olive oil'}), confidence=0.06944444444444445, lift=6.128267973856209), OrderedStatistic(items_base=frozenset({'olive oil', 'milk'}), items_add=frozenset({'frozen vegetables', 'mineral water'}), confidence=0.19531250000000003, lift=5.4665636660447765), OrderedStatistic(items_base=frozenset({'mineral water', 'olive oil'}), items_add=frozenset({'frozen vegetables', 'milk'}), confidence=0.12077294685990338, lift=5.118180081334097), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water', 'milk'}), items_add=frozenset({'olive oil'}), confidence=0.30120481927710846, lift=4.573557387444516), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'olive oil', 'mineral water'}), items_add=frozenset({'milk'}), confidence=0.5102040816326531, lift=3.937284790459394), OrderedStatistic(items_base=frozenset({'mineral water', 'olive oil', 'milk'}), items_add=frozenset({'frozen vegetables'}), confidence=0.39062500000000006, lift=4.098011363636364)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'soup', 'mineral water', 'milk'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'mineral water', 'soup', 'milk'}), confidence=0.03216783216783217, lift=3.7701704545454553), OrderedStatistic(items_base=frozenset({'milk'}), items_add=frozenset({'frozen vegetables', 'soup', 'mineral water'}), confidence=0.02366255144032922, lift=4.670863114576566), OrderedStatistic(items_base=frozenset({'mineral water'}), items_add=frozenset({'frozen vegetables', 'soup', 'milk'}), confidence=0.012863534675615214, lift=3.2163124533929905), OrderedStatistic(items_base=frozenset({'soup'}), items_add=frozenset({'frozen vegetables', 'mineral water', 'milk'}), confidence=0.06068601583113457, lift=5.484407286136632), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'milk'}), items_add=frozenset({'mineral water', 'soup'}), confidence=0.12994350282485875, lift=5.634139969302113), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water'}), items_add=frozenset({'soup', 'milk'}), confidence=0.08582089552238806, lift=5.646864362398533), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'soup'}), items_add=frozenset({'mineral water', 'milk'}), confidence=0.3833333333333333, lift=7.987175925925926), OrderedStatistic(items_base=frozenset({'mineral water', 'milk'}), items_add=frozenset({'frozen vegetables', 'soup'}), confidence=0.0638888888888889, lift=7.987175925925926), OrderedStatistic(items_base=frozenset({'soup', 'milk'}), items_add=frozenset({'frozen vegetables', 'mineral water'}), confidence=0.20175438596491227, lift=5.646864362398533), OrderedStatistic(items_base=frozenset({'mineral water', 'soup'}), items_add=frozenset({'frozen vegetables', 'milk'}), confidence=0.13294797687861273, lift=5.6341399693021135), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water', 'milk'}), items_add=frozenset({'soup'}), confidence=0.27710843373493976, lift=5.484407286136631), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'soup', 'milk'}), items_add=frozenset({'mineral water'}), confidence=0.7666666666666666, lift=3.21631245339299), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'soup', 'mineral water'}), items_add=frozenset({'milk'}), confidence=0.6052631578947368, lift=4.670863114576565), OrderedStatistic(items_base=frozenset({'mineral water', 'soup', 'milk'}), items_add=frozenset({'frozen vegetables'}), confidence=0.35937500000000006, lift=3.7701704545454553)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'spaghetti', 'mineral water', 'milk'}), support=0.004532728969470737, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'mineral water', 'spaghetti', 'milk'}), confidence=0.04755244755244755, lift=3.0228043143297376), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'milk'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.19209039548022597, lift=3.216227804681194), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water'}), items_add=frozenset({'spaghetti', 'milk'}), confidence=0.12686567164179102, lift=3.5775165525754677), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti'}), items_add=frozenset({'mineral water', 'milk'}), confidence=0.16267942583732056, lift=3.389606592238171), OrderedStatistic(items_base=frozenset({'mineral water', 'milk'}), items_add=frozenset({'frozen vegetables', 'spaghetti'}), confidence=0.09444444444444444, lift=3.389606592238171), OrderedStatistic(items_base=frozenset({'spaghetti', 'milk'}), items_add=frozenset({'frozen vegetables', 'mineral water'}), confidence=0.12781954887218044, lift=3.577516552575468), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'frozen vegetables', 'milk'}), confidence=0.07589285714285714, lift=3.216227804681194), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti', 'milk'}), items_add=frozenset({'frozen vegetables'}), confidence=0.28813559322033894, lift=3.0228043143297376)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'spaghetti', 'mineral water', 'shrimp'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'mineral water', 'spaghetti', 'shrimp'}), confidence=0.03496503496503497, lift=4.098011363636364), OrderedStatistic(items_base=frozenset({'shrimp'}), items_add=frozenset({'frozen vegetables', 'spaghetti', 'mineral water'}), confidence=0.046641791044776115, lift=3.8873341625207294), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water'}), items_add=frozenset({'spaghetti', 'shrimp'}), confidence=0.09328358208955223, lift=4.400755655683844), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'shrimp'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.2, lift=3.3486607142857148), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti'}), items_add=frozenset({'mineral water', 'shrimp'}), confidence=0.11961722488038279, lift=5.069202281512719), OrderedStatistic(items_base=frozenset({'mineral water', 'shrimp'}), items_add=frozenset({'frozen vegetables', 'spaghetti'}), confidence=0.14124293785310735, lift=5.069202281512719), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'frozen vegetables', 'shrimp'}), confidence=0.05580357142857143, lift=3.3486607142857143), OrderedStatistic(items_base=frozenset({'spaghetti', 'shrimp'}), items_add=frozenset({'frozen vegetables', 'mineral water'}), confidence=0.15723270440251574, lift=4.400755655683845), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti', 'mineral water'}), items_add=frozenset({'shrimp'}), confidence=0.2777777777777778, lift=3.8873341625207294), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti', 'shrimp'}), items_add=frozenset({'frozen vegetables'}), confidence=0.39062500000000006, lift=4.098011363636364)]),
     RelationRecord(items=frozenset({'frozen vegetables', 'spaghetti', 'mineral water', 'tomatoes'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'frozen vegetables'}), items_add=frozenset({'mineral water', 'spaghetti', 'tomatoes'}), confidence=0.03216783216783217, lift=3.4470129870129873), OrderedStatistic(items_base=frozenset({'spaghetti'}), items_add=frozenset({'frozen vegetables', 'mineral water', 'tomatoes'}), confidence=0.01761102603369066, lift=3.0022796881525826), OrderedStatistic(items_base=frozenset({'tomatoes'}), items_add=frozenset({'frozen vegetables', 'spaghetti', 'mineral water'}), confidence=0.04483430799220273, lift=3.736690491661252), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water'}), items_add=frozenset({'spaghetti', 'tomatoes'}), confidence=0.08582089552238806, lift=4.1002709383021205), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti'}), items_add=frozenset({'mineral water', 'tomatoes'}), confidence=0.11004784688995216, lift=4.510759013778858), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'tomatoes'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.1900826446280992, lift=3.1826114226682414), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'frozen vegetables', 'tomatoes'}), confidence=0.05133928571428572, lift=3.1826114226682414), OrderedStatistic(items_base=frozenset({'mineral water', 'tomatoes'}), items_add=frozenset({'frozen vegetables', 'spaghetti'}), confidence=0.12568306010928962, lift=4.510759013778858), OrderedStatistic(items_base=frozenset({'spaghetti', 'tomatoes'}), items_add=frozenset({'frozen vegetables', 'mineral water'}), confidence=0.14649681528662423, lift=4.1002709383021205), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'spaghetti', 'mineral water'}), items_add=frozenset({'tomatoes'}), confidence=0.2555555555555556, lift=3.7366904916612524), OrderedStatistic(items_base=frozenset({'frozen vegetables', 'mineral water', 'tomatoes'}), items_add=frozenset({'spaghetti'}), confidence=0.5227272727272727, lift=3.0022796881525826), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti', 'tomatoes'}), items_add=frozenset({'frozen vegetables'}), confidence=0.32857142857142857, lift=3.447012987012987)]),
     RelationRecord(items=frozenset({'ground beef', 'mineral water', 'spaghetti', 'milk'}), support=0.004399413411545127, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef', 'milk'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.2, lift=3.3486607142857148), OrderedStatistic(items_base=frozenset({'ground beef', 'mineral water'}), items_add=frozenset({'spaghetti', 'milk'}), confidence=0.10749185667752444, lift=3.0311895373613185), OrderedStatistic(items_base=frozenset({'spaghetti', 'milk'}), items_add=frozenset({'ground beef', 'mineral water'}), confidence=0.12406015037593984, lift=3.0311895373613185), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'ground beef', 'milk'}), confidence=0.07366071428571429, lift=3.3486607142857148)]),
     RelationRecord(items=frozenset({'ground beef', 'mineral water', 'olive oil', 'spaghetti'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'spaghetti', 'mineral water', 'olive oil'}), confidence=0.03120759837177748, lift=3.040106433593544), OrderedStatistic(items_base=frozenset({'ground beef', 'mineral water'}), items_add=frozenset({'spaghetti', 'olive oil'}), confidence=0.0749185667752443, lift=3.267233542913416), OrderedStatistic(items_base=frozenset({'ground beef', 'olive oil'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.2169811320754717, lift=3.63298096361186), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'ground beef', 'olive oil'}), confidence=0.05133928571428572, lift=3.63298096361186), OrderedStatistic(items_base=frozenset({'spaghetti', 'olive oil'}), items_add=frozenset({'ground beef', 'mineral water'}), confidence=0.13372093023255816, lift=3.267233542913416), OrderedStatistic(items_base=frozenset({'spaghetti', 'mineral water', 'olive oil'}), items_add=frozenset({'ground beef'}), confidence=0.2987012987012987, lift=3.0401064335935435)]),
     RelationRecord(items=frozenset({'ground beef', 'mineral water', 'spaghetti', 'pancakes'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef', 'pancakes'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.21100917431192662, lift=3.532990661861075), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'ground beef', 'pancakes'}), confidence=0.05133928571428572, lift=3.532990661861075)]),
     RelationRecord(items=frozenset({'ground beef', 'mineral water', 'spaghetti', 'tomatoes'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'ground beef'}), items_add=frozenset({'mineral water', 'spaghetti', 'tomatoes'}), confidence=0.03120759837177748, lift=3.344117076952898), OrderedStatistic(items_base=frozenset({'spaghetti'}), items_add=frozenset({'ground beef', 'mineral water', 'tomatoes'}), confidence=0.01761102603369066, lift=3.221958689724723), OrderedStatistic(items_base=frozenset({'ground beef', 'mineral water'}), items_add=frozenset({'spaghetti', 'tomatoes'}), confidence=0.0749185667752443, lift=3.579389613892405), OrderedStatistic(items_base=frozenset({'ground beef', 'spaghetti'}), items_add=frozenset({'mineral water', 'tomatoes'}), confidence=0.0782312925170068, lift=3.2066280063938146), OrderedStatistic(items_base=frozenset({'ground beef', 'tomatoes'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.26136363636363635, lift=4.3760907061688314), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'ground beef', 'tomatoes'}), confidence=0.05133928571428572, lift=4.3760907061688314), OrderedStatistic(items_base=frozenset({'mineral water', 'tomatoes'}), items_add=frozenset({'ground beef', 'spaghetti'}), confidence=0.12568306010928962, lift=3.206628006393814), OrderedStatistic(items_base=frozenset({'spaghetti', 'tomatoes'}), items_add=frozenset({'ground beef', 'mineral water'}), confidence=0.14649681528662423, lift=3.579389613892405), OrderedStatistic(items_base=frozenset({'ground beef', 'mineral water', 'tomatoes'}), items_add=frozenset({'spaghetti'}), confidence=0.5609756097560976, lift=3.221958689724723), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti', 'tomatoes'}), items_add=frozenset({'ground beef'}), confidence=0.32857142857142857, lift=3.344117076952898)]),
     RelationRecord(items=frozenset({'spaghetti', 'mineral water', 'olive oil', 'milk'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'olive oil'}), items_add=frozenset({'mineral water', 'spaghetti', 'milk'}), confidence=0.05060728744939272, lift=3.216993755575379), OrderedStatistic(items_base=frozenset({'mineral water', 'milk'}), items_add=frozenset({'olive oil', 'spaghetti'}), confidence=0.06944444444444445, lift=3.0285045219638245), OrderedStatistic(items_base=frozenset({'olive oil', 'milk'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.19531250000000003, lift=3.2701764787946432), OrderedStatistic(items_base=frozenset({'spaghetti', 'milk'}), items_add=frozenset({'mineral water', 'olive oil'}), confidence=0.09398496240601503, lift=3.4057062947223127), OrderedStatistic(items_base=frozenset({'mineral water', 'olive oil'}), items_add=frozenset({'spaghetti', 'milk'}), confidence=0.12077294685990338, lift=3.4057062947223127), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'olive oil', 'milk'}), confidence=0.05580357142857143, lift=3.2701764787946432), OrderedStatistic(items_base=frozenset({'spaghetti', 'olive oil'}), items_add=frozenset({'mineral water', 'milk'}), confidence=0.14534883720930233, lift=3.0285045219638245), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti', 'milk'}), items_add=frozenset({'olive oil'}), confidence=0.211864406779661, lift=3.216993755575379)]),
     RelationRecord(items=frozenset({'mineral water', 'spaghetti', 'shrimp', 'milk'}), support=0.0030662578322890282, ordered_statistics=[OrderedStatistic(items_base=frozenset({'mineral water', 'milk'}), items_add=frozenset({'spaghetti', 'shrimp'}), confidence=0.0638888888888889, lift=3.014028651292803), OrderedStatistic(items_base=frozenset({'spaghetti', 'milk'}), items_add=frozenset({'mineral water', 'shrimp'}), confidence=0.08646616541353383, lift=3.6643090777791936), OrderedStatistic(items_base=frozenset({'mineral water', 'shrimp'}), items_add=frozenset({'spaghetti', 'milk'}), confidence=0.12994350282485875, lift=3.6643090777791936), OrderedStatistic(items_base=frozenset({'spaghetti', 'shrimp'}), items_add=frozenset({'mineral water', 'milk'}), confidence=0.14465408805031446, lift=3.014028651292802)]),
     RelationRecord(items=frozenset({'mineral water', 'spaghetti', 'tomatoes', 'milk'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'tomatoes'}), items_add=frozenset({'mineral water', 'spaghetti', 'milk'}), confidence=0.04873294346978558, lift=3.097845838702217), OrderedStatistic(items_base=frozenset({'mineral water', 'milk'}), items_add=frozenset({'spaghetti', 'tomatoes'}), confidence=0.06944444444444445, lift=3.3178520877565467), OrderedStatistic(items_base=frozenset({'spaghetti', 'milk'}), items_add=frozenset({'mineral water', 'tomatoes'}), confidence=0.09398496240601503, lift=3.8523563005875343), OrderedStatistic(items_base=frozenset({'tomatoes', 'milk'}), items_add=frozenset({'mineral water', 'spaghetti'}), confidence=0.2380952380952381, lift=3.9865008503401365), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti'}), items_add=frozenset({'milk', 'tomatoes'}), confidence=0.05580357142857143, lift=3.9865008503401365), OrderedStatistic(items_base=frozenset({'mineral water', 'tomatoes'}), items_add=frozenset({'spaghetti', 'milk'}), confidence=0.1366120218579235, lift=3.8523563005875348), OrderedStatistic(items_base=frozenset({'spaghetti', 'tomatoes'}), items_add=frozenset({'mineral water', 'milk'}), confidence=0.15923566878980894, lift=3.3178520877565467), OrderedStatistic(items_base=frozenset({'mineral water', 'spaghetti', 'milk'}), items_add=frozenset({'tomatoes'}), confidence=0.211864406779661, lift=3.0978458387022165)])]




```python
df_results = pd.DataFrame(Results)
df_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>items</th>
      <th>support</th>
      <th>ordered_statistics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(brownies, cottage cheese)</td>
      <td>0.003466</td>
      <td>[((brownies), (cottage cheese), 0.102766798418...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(chicken, light cream)</td>
      <td>0.004533</td>
      <td>[((chicken), (light cream), 0.0755555555555555...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(escalope, mushroom cream sauce)</td>
      <td>0.005733</td>
      <td>[((escalope), (mushroom cream sauce), 0.072268...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(pasta, escalope)</td>
      <td>0.005866</td>
      <td>[((escalope), (pasta), 0.07394957983193277, 4....</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(fresh bread, tomato juice)</td>
      <td>0.004266</td>
      <td>[((fresh bread), (tomato juice), 0.09907120743...</td>
    </tr>
  </tbody>
</table>
</div>




```python
support = df_results.support
first_values = []
second_values = []
third_values = []
fourth_value = []
for i in range(df_results.shape[0]):
    single_list = df_results['ordered_statistics'][i][0]
    first_values.append(list(single_list[0]))
    second_values.append(list(single_list[1]))
    third_values.append(single_list[2])
    fourth_value.append(single_list[3])
lhs = pd.DataFrame(first_values)
rhs= pd.DataFrame(second_values)
confidance=pd.DataFrame(third_values,columns=['Confidance'])
lift=pd.DataFrame(fourth_value,columns=['lift'])
df_final = pd.concat([lhs,rhs,support,confidance,lift], axis=1)
df_final
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>support</th>
      <th>Confidance</th>
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>brownies</td>
      <td>None</td>
      <td>cottage cheese</td>
      <td>None</td>
      <td>None</td>
      <td>0.003466</td>
      <td>0.102767</td>
      <td>3.225330</td>
    </tr>
    <tr>
      <th>1</th>
      <td>chicken</td>
      <td>None</td>
      <td>light cream</td>
      <td>None</td>
      <td>None</td>
      <td>0.004533</td>
      <td>0.075556</td>
      <td>4.843951</td>
    </tr>
    <tr>
      <th>2</th>
      <td>escalope</td>
      <td>None</td>
      <td>mushroom cream sauce</td>
      <td>None</td>
      <td>None</td>
      <td>0.005733</td>
      <td>0.072269</td>
      <td>3.790833</td>
    </tr>
    <tr>
      <th>3</th>
      <td>escalope</td>
      <td>None</td>
      <td>pasta</td>
      <td>None</td>
      <td>None</td>
      <td>0.005866</td>
      <td>0.073950</td>
      <td>4.700812</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fresh bread</td>
      <td>None</td>
      <td>tomato juice</td>
      <td>None</td>
      <td>None</td>
      <td>0.004266</td>
      <td>0.099071</td>
      <td>3.259356</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>89</th>
      <td>ground beef</td>
      <td>pancakes</td>
      <td>mineral water</td>
      <td>spaghetti</td>
      <td>None</td>
      <td>0.003066</td>
      <td>0.211009</td>
      <td>3.532991</td>
    </tr>
    <tr>
      <th>90</th>
      <td>ground beef</td>
      <td>None</td>
      <td>mineral water</td>
      <td>spaghetti</td>
      <td>tomatoes</td>
      <td>0.003066</td>
      <td>0.031208</td>
      <td>3.344117</td>
    </tr>
    <tr>
      <th>91</th>
      <td>olive oil</td>
      <td>None</td>
      <td>mineral water</td>
      <td>spaghetti</td>
      <td>milk</td>
      <td>0.003333</td>
      <td>0.050607</td>
      <td>3.216994</td>
    </tr>
    <tr>
      <th>92</th>
      <td>mineral water</td>
      <td>milk</td>
      <td>spaghetti</td>
      <td>shrimp</td>
      <td>None</td>
      <td>0.003066</td>
      <td>0.063889</td>
      <td>3.014029</td>
    </tr>
    <tr>
      <th>93</th>
      <td>tomatoes</td>
      <td>None</td>
      <td>mineral water</td>
      <td>spaghetti</td>
      <td>milk</td>
      <td>0.003333</td>
      <td>0.048733</td>
      <td>3.097846</td>
    </tr>
  </tbody>
</table>
<p>94 rows × 8 columns</p>
</div>




```python
df_final.pop(1)
df_final.pop(2)
df_final.head()
df_final.head()
df_final.columns = ['lhs','rhs','support','confidance','lift'] 
df_final['lhs'] = df_final['lhs']+str(", ") 

df_final
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lhs</th>
      <th>rhs</th>
      <th>support</th>
      <th>confidance</th>
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>brownies,</td>
      <td>cottage cheese</td>
      <td>0.003466</td>
      <td>0.102767</td>
      <td>3.225330</td>
    </tr>
    <tr>
      <th>1</th>
      <td>chicken,</td>
      <td>light cream</td>
      <td>0.004533</td>
      <td>0.075556</td>
      <td>4.843951</td>
    </tr>
    <tr>
      <th>2</th>
      <td>escalope,</td>
      <td>mushroom cream sauce</td>
      <td>0.005733</td>
      <td>0.072269</td>
      <td>3.790833</td>
    </tr>
    <tr>
      <th>3</th>
      <td>escalope,</td>
      <td>pasta</td>
      <td>0.005866</td>
      <td>0.073950</td>
      <td>4.700812</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fresh bread,</td>
      <td>tomato juice</td>
      <td>0.004266</td>
      <td>0.099071</td>
      <td>3.259356</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>89</th>
      <td>ground beef,</td>
      <td>mineral water</td>
      <td>0.003066</td>
      <td>0.211009</td>
      <td>3.532991</td>
    </tr>
    <tr>
      <th>90</th>
      <td>ground beef,</td>
      <td>mineral water</td>
      <td>0.003066</td>
      <td>0.031208</td>
      <td>3.344117</td>
    </tr>
    <tr>
      <th>91</th>
      <td>olive oil,</td>
      <td>mineral water</td>
      <td>0.003333</td>
      <td>0.050607</td>
      <td>3.216994</td>
    </tr>
    <tr>
      <th>92</th>
      <td>mineral water,</td>
      <td>spaghetti</td>
      <td>0.003066</td>
      <td>0.063889</td>
      <td>3.014029</td>
    </tr>
    <tr>
      <th>93</th>
      <td>tomatoes,</td>
      <td>mineral water</td>
      <td>0.003333</td>
      <td>0.048733</td>
      <td>3.097846</td>
    </tr>
  </tbody>
</table>
<p>94 rows × 5 columns</p>
</div>



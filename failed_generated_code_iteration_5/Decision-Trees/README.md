# Decison Trees
This projects aims to create supervised machine learning with decision trees. A csv file is added for exploration.

## How it Works
The user provides a CSV file into the local directory. The program then parses through the csv and selects the best attribute to make a split on. This is done recursively for each level of the tree until leaf nodes are created.

An attribute would correspond to a node in the tree and the split represents the two different paths that can be taken given a threshold of that attribute. 

The best attribute to split is based on the convept of information gain. Information gain is an equation that depends on the amount of entropy in set of examples generates by the split. Entropy is how homogenous the classifcations are in a set of examples. The lower the entropy, the higher the amount of unified classification.

Leaf nodes are created following a threshold criteria, provided at the start of the program, called the min_leaf_count. This basically means every nodes and its corresponding set of examples must have a size >= the min_leaf_count. Thus if we try and split node D further, but every split leads to split sizes smaller than this threshold than the original node D would become a leaf node.

## Metrics
The model takes a percentage of the model data for learning and the rest for testing. 

When the model is tested, the accuracy score will be printed (as a percentage) and so will the "almost" score, which represents how many near correct classication were made of the total.

A confusion matrix will also be printed with the model of the tree itself and the attributes and thresholds.

"""
Peter Robe
Decision Tree
"""

import math
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union, Dict, Any


def is_number(value):
    return isinstance(value, int) or isinstance(value, float)


class Decision:
    """
    A decision used to branch in a tree. It contains the column_name and a value.
    If the value is a number, it will decide whether a new value is greater than or equal to the decision's value.
    If it's anything else (categorical), it will decide whether the new value is equal to the decision's value
    """
    def __init__(self, value, column_name: str):
        self.value = value
        self.column_name = column_name

    def is_true(self, input_features: pd.Series):
        """
        A function that decides whether a row of features matches the decision
        :param input_features: pandas Series
        :return: cool
        """
        if is_number(input_features[self.column_name]):
            return self.value >= input_features[self.column_name]
        else:
            return self.value == input_features[self.column_name]

    def __str__(self):
        if is_number(self.value):
            return "Is column " + str(self.column_name) + " >= " + str(self.value) + "?"
        else:
            return "Is column " + str(self.column_name) + " == " + str(self.value) + "?"


class Leaf:
    """
    A leaf node of a decision tree
    Contains a dictionary of all the members of the leaf node with the number of occurrences
    """

    def __init__(self, prediction_dict: Dict[Any, int]):
        self.prediction_dict: Dict[Any, int] = prediction_dict

    def most_predicted(self):
        """
       A method to return the most occurring value in a occurrences dictionary
        :return Any predicted value
        """
        v = list(self.prediction_dict.values())
        k = list(self.prediction_dict.keys())
        return k[v.index(max(v))]


class Branch:
    """
    A normal node in the decision tree
    Contains a decision that breaks the tree into two separate branches
    """

    def __init__(self, decision: Decision, true_branch: Union[Leaf, 'Branch'], false_branch: Union[Leaf, 'Branch']):
        self.decision = decision
        self.true_branch = true_branch
        self.false_branch = false_branch


class Tree:
    """
    A Tree holds a reference to the root node. It will train the decision tree and can make classifications after
    it's trained.
    """

    def __init__(self, data: pd.DataFrame, label_name):
        # The column name of the label
        self.label_name = label_name
        # Based on the command line argument, which function determines uncertainty
        if sys.argv[1] == "gini":
            self.uncertainty_fun = self.gini
        else:
            self.uncertainty_fun = self.entropy
        # Build the decision tree using training data
        self.root_node = self.build_tree(data)

    def gini(self, df: pd.DataFrame) -> float:
        """
        Calculates the Gini Index for a DataFrame
        :param df pandas DataFrame
        :return int Gini Index value
        """
        counts_dict = self.classification_counts(df)
        impurity = 1
        for key in counts_dict:
            proportion = counts_dict[key] / len(df.index)
            impurity -= proportion ** 2
        return impurity

    def entropy(self, df: pd.DataFrame) -> float:
        """
        Determines the Entropy for a DataFrame
        :param df: pandas DataFrame
        :return float Entropy value
        """
        counts_dict = self.classification_counts(df)
        impurity = 0
        for key in counts_dict:
            proportion = counts_dict[key] / len(df.index)
            impurity += proportion * math.log2(proportion)
        return -impurity

    def branch_uncertainty(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """
        Calculates the proportionally weighted uncertainty of two new branches
        :param df1: pandas DataFrame (left branch)
        :param df2: pandas DataFrame (right branch)
        :return: float combined uncertainty value of these branches
        """
        proportion = len(df1.index) / (len(df1.index) + len(df2.index))
        return proportion * self.uncertainty_fun(df1) + (1 - proportion) * self.uncertainty_fun(df2)

    def best_decision(self, data: pd.DataFrame):
        """
        Determines the decision that best minimizes uncertainty (most information gain)
        :param data: pandas DataFrame
        :return: Decision the best decision, None if no decision provides more information gain (leaf)
        """
        current_uncertainty = self.uncertainty_fun(data)

        best_info_gain = 0
        best_decision = None

        # Drop the label column so decisions can only be about the features
        features = data.drop(self.label_name, axis='columns')

        # For every column of the features
        for column_name, column in features.iteritems():
            # For every unique value of this feature
            for value in column.unique():
                # Create a potential decision based off this value
                potential_decision = Decision(value, column_name)

                # Split the data based on this decision
                true_rows, false_rows = self.predictions(data, potential_decision)

                # Ignore this decision if it doesn't split the data at all
                if true_rows.empty or false_rows.empty:
                    continue

                # Calculate the combined uncertainty of the branches
                new_uncertainty = self.branch_uncertainty(true_rows, false_rows)

                # Calculate the information gain
                info_gain = current_uncertainty - new_uncertainty

                # If this decision has the most information gain so far, it's the new best decision
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_decision = potential_decision
        return best_decision

    def classification_counts(self, data: pd.DataFrame) -> Dict[Any, int]:
        """
        Returns a dictionary that maps each unique value of the label with the number of occurrences
        { "value" -> "num occurrences", ... }
        :param data: pandas DataFrame
        :return: occurrences dictionary
        """
        label_counts: Dict[Any, int] = data[self.label_name].value_counts().to_dict()
        return label_counts

    @staticmethod
    def predictions(df: pd.DataFrame, decision: Decision) -> (pd.DataFrame, pd.DataFrame):
        """
        Split the data into two separate DataFrames based on a Decision
        :param df: pandas DataFrame
        :param decision: Decision to split on
        :return: (DataFrame, DataFrame)
        """
        mask = df.apply(lambda x: decision.is_true(x), axis='columns')
        true_df = df[mask]
        false_df = df[~mask]
        return true_df, false_df

    def build_tree(self, data: pd.DataFrame) -> Union[Leaf, Branch]:
        """
        Recursively builds the rest of the tree
        :param data: The data that we want to divide
        :return: Either a Leaf or a Branch depending on whether a decision leads to information gain
        """

        # Calculate the best decision to split the data
        decision = self.best_decision(data)

        # Return a Leaf if no decision gains information
        if decision is None:
            # The Leaf contains a dictionary that maps the occurrences of the label values at this point
            return Leaf(self.classification_counts(data))

        # Split the data on the decision
        true_df, false_df = self.predictions(data, decision)

        # Recursively build out the tree from the new branches
        true_branch = self.build_tree(true_df)
        false_branch = self.build_tree(false_df)

        # Return this new branch made from the decision and the two complete child branches
        return Branch(decision, true_branch, false_branch)

    def classify(self, features: pd.Series) -> Any:
        """
        Public method to classify a row of features. If the tree hasn't been built yet, return None
        :param features: pandas Series
        :return: predicted label value
        """
        if self.root_node is None:
            print("Not trained yet!")
            return None
        else:
            return self.__rec_classify(self.root_node, features)

    def __rec_classify(self, node: Union[Leaf, Branch], features: pd.Series) -> Any:
        """
        The recursive call for classifying a row of features. Traverses the tree depending on how the decision at the current node
        :param node: Node the current node in the tree
        :param features: the row of features to decide on
        :return: The prediction value
        """
        if isinstance(node, Leaf):
            return node.most_predicted()
        else:
            if node.decision.is_true(features):
                return self.__rec_classify(node.true_branch, features)
            else:
                return self.__rec_classify(node.false_branch, features)


##
# Runs a decision tree algorithm on a data file
# @param filename String The name of the file to read from that must be located in the same directory
# @param delimit String That is the delimiter for the items in a row
# @param label_col int Which column (starting from 0) is the one with the label
# @param remove_col array[int] The columns to remove
# @param header bool Whether or not there is a header
# @return float The accuracy of the k-NN prediction algorithm
#
def decide(filename, delimit, label_col, remove_col, header):
    # A Pandas DataFrame with all the data. Both training and testing
    data = pd.DataFrame()
    # Removes the first row if the columns are labeled
    if header:
        data = pd.read_csv(filename, delimiter=delimit, index_col=False)
    else:
        data = pd.read_csv(filename, delimiter=delimit, header=None, index_col=False)
    # Removes any rows that are not wanted
    if len(remove_col) > 0:
        data = data.drop(remove_col, axis=1)

        for col_num in remove_col:
            if label_col > col_num:
                label_col -= 1
            elif label_col == col_num:
                print("Cannot Delete the Label Column")

    # Split the data into train/test
    train, test = train_test_split(data, shuffle=False)

    # Build the Tree
    tree = Tree(train, label_col)

    # Calculate the total accuracy by predicting the testing data
    accuracies = 0
    for index, item in test.iterrows():
        if item[label_col] == tree.classify(item):
            accuracies += 1
    accuracies /= len(test.index)
    # print("Actual: " + str(item[label_col]))
    # print("Prediction: " + str(tree.classify(item)) + "\n")
    return accuracies


if __name__ == "__main__":
    breast_accuracy = decide("breast-cancer-wisconsin.data", ",", 10, [0], False)
    print("Breast Accuracy: " + str(breast_accuracy))
    test_accuracy = decide("winequality-red.csv", ';', "quality", [], True)
    print("Wine Accuracy: " + str(test_accuracy))

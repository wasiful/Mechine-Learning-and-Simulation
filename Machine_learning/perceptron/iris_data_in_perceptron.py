import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Machine_learning.perceptron.perceptron_method import Perceptron
from matplotlib.colors import ListedColormap

#          checking current directory
# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))
# Change the current working directory
# os.chdir("/home/psycastro/PycharmProjects/MlandSim/Machine_learning/perceptron/iris_data_in_perceptron.py")

#         loading data
df = pd.read_csv('Machine_learning/all_ml_data/iris.data', header=None, encoding='utf-8')
df.tail()

#     checking data plot from irish data
y = df.iloc[0:100, 4].values
print(f'y = {y}')
y = np.where(y == 'Iris-setosa', 0, 1)
print(f'y where = {y}')
x = df.iloc[0:100, [0, 2]].values
print(f'x = {x}')

plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='s', label='versicolor')
plt.xlabel('Sepal length[cm]')
plt.ylabel('petal length[cm]')
plt.legend(loc='upper left')
plt.show()

#    using the perceptron method - misclassified errors against number of epochs
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()


#    visualizing decision boundaries
def plot_decision_regions(x, y, classifier, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    print(f'x1min = {x1_min}\n, x1max = {x1_max}\n, x2min = {x2_min}\n, x2min = {x2_max}\n'
          f', xx1 = {xx1}\n, xx2 = {xx2}\n, lab = {lab}\n')

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=x[y == c1, 0], y = x[y == c1, 1], alpha=0.8, label=f'class{c1}', edgecolor='black')


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

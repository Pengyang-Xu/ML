from sklearn.datasets import fetch_openml
import numpy as np
print('ok')
def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target,i) for i,target in enumerate(mnist.target[:60000])]))[:,1]  #训练集
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:,1]  #测试集
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]
try:
    mnist = fetch_openml('mnist_784',version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
    sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
except:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]
print(X.shape)
print('ok')
#encoding:utf8
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn
from sklearn import datasets, cross_validation

# feature 数据的特征
# target 数据每一行的目标或者分类的标识
def my_model(features, target):
    target = tf.one_hot(target, 3, 1, 0)
    features = layers.stack(features, layers.fully_connected, [10, 20, 10])
    prediction, loss = tf.contrib.learn.models.logistic_regression_zero_init(features, target)
    train_op = tf.contrib.layers.optmize_loss(loss, tf.contrib.framework.get_global_step(),
                                              optimizer='Adagrad', learning_rate=0.1)
    return {'class': tf.arg_max(prediction, 1), 'prob': prediction}, loss, train_op

if __name__ == "__main__":
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=35)

    classifer = learn.Estimator(model_fn=my_model)
    classifer.fit(x_train, y_train, steps=700)

    predictions = classifer.predict(x_test)

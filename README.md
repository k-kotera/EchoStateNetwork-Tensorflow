# EchoStateNetwork-Tensorflow
Minimal and Simple Tensorflow implementation of EchoStateNetwork
### short introduction
```python
#Run ESN
with tf.Session() as example_sess:
    ESN = EchoStateNetwork(example_sess, units=32)
    ESN.fit(X_train,y_train)
    y_test_hat = ESN.predict(X_test)
```
![result](https://raw.githubusercontent.com/k-kotera/EchoStateNetwork-Tensorflow/master/images/result.png)

# EchoStateNetwork-Tensorflow
Minimal and Simple Tensorflow implementation of EchoStateNetwork
### short introduction
You can easily use this like "scikit-learn".
```python
ESN = EchoStateNetwork(units=32)
ESN.fit(X_train, y_train)
y_test_hat = ESN.predict(X_test)

print("Train MSE:",ESN.MSE_Score(X_train, y_train))
print("Test MSE:",ESN.MSE_Score(X_test, y_test))
```
![result](https://raw.githubusercontent.com/k-kotera/EchoStateNetwork-Tensorflow/master/images/result.png)

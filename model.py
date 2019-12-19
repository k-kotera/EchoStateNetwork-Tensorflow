import tensorflow as tf
import numpy as np

def Tikhonov_regularization(x,y,beta=0.01):    
    """
    tensorflow implementation of Ridge regression (also known as Tikhonov regularization)

    Parameters
    x : input data with shape (batch_size, input_dim)
    y : target data with shape (batch_size, output_dim)
    Returns
    _W_out_estimated : estimated weight with shape (input_dim, output_dim)
    """
    
    X = tf.constant(x, tf.float32)
    Y = tf.constant(y, tf.float32) # Y:[T,Ny]

    Xt_X = tf.matmul(tf.transpose(X),X)
    Xt_X_plus_betaI = Xt_X + beta * tf.eye(int(X.shape[1]))
    Xt_Y = tf.matmul(tf.transpose(X),Y)
    W_out_estimated = tf.matmul(tf.linalg.inv(Xt_X_plus_betaI), Xt_Y)

    with tf.Session() as _sess:
        _W_out_estimated = _sess.run(W_out_estimated)
    
    return _W_out_estimated

class ESNCell(tf.keras.layers.Layer):
    """
    ESN single cell for "tf.keras.layers.RNN".
    """
    
    def __init__(self, units, sr_scale=1.0, density=0.2, leaking_rate=0.9, **kwargs):
        
        def _W_initializer(shape, dtype=None, partition_info=None):
            w_init = tf.random.normal(shape, dtype=dtype)
            mask = tf.cast(tf.math.less_equal(tf.random.uniform(shape), self._density), dtype) #sparse 0-1 matrix
            w_init_sparse = w_init * mask
            Eigenvalues_w_init, Eigenvectors_w_init = tf.linalg.eigh(w_init_sparse)
            Spectral_radius = tf.math.reduce_max(tf.abs(Eigenvalues_w_init))
            w_init_sparse_r = w_init_sparse * self._sr_scale / Spectral_radius #normalization based on Spectral_radius
            return w_init_sparse_r
        
        self.state_size = units
        self._sr_scale = sr_scale
        self._density = density
        self._leaking_rate = leaking_rate
        self._W_initializer = _W_initializer
        super(ESNCell, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W_in = self.add_weight(shape= (input_shape[-1], self.state_size),
                                    initializer=tf.random_normal_initializer,
                                    trainable = False,
                                    name='W_in')
        self.W = self.add_weight(shape=(self.state_size, self.state_size),
                                 initializer=self._W_initializer,
                                 trainable = False,
                                 name='W')
        self.b = self.add_weight(shape=(self.state_size,),
                                 initializer=tf.random_normal_initializer,
                                 trainable = False,
                                 name='Bias')
        self.built = True

    def call(self, inputs, states):
        x_n_1 = states[0] #x[n-1]
        x_tilda_n = tf.math.tanh(tf.tensordot(inputs, self.W_in, axes=1) + tf.tensordot(x_n_1, self.W, axes=1) + self.b)
        x_n = (1 - self._leaking_rate) * x_n_1 + self._leaking_rate * x_tilda_n
        return x_n, [x_n]

class EchoStateNetwork():
    """
    Echo State Network for regression.

    Parameters
    sess : specify the tensorflow session.
    units : dimensionality of the reservoir unit (int, default=30)
    sr_scale : scale for spectral radius of W (float, default=1.0)
               sr_scale < 1.0 ensures echo state property in most situations.
               But empirically you can specify it over 1.0.
    density : density of W sparsity (float, default=0.2)
    leaking_rate : leaking rate α ∈ (0, 1]  (float, default=0.9)
    """
    
    def __init__(self, sess=None, units=30, sr_scale=1.0, density=0.2, leaking_rate=0.9):
        if sess == None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        self.units = units
        self.sr_scale = sr_scale
        self.density = density
        self.leaking_rate = leaking_rate
        self.trained = False

    def fit(self, x, y, beta=0.01):     
        """
        build and fit the model according to the given training data.
        
        Parameters
        x : input data with shape (batch_size, timesteps, input_dim)
        y : target data with shape (batch_size, output_dim)
        beta : Ridge regression parameter (float, default=0.01)
        """
        
        self.InputShape = x.shape
        self.OutputShape = y.shape
        if len(self.InputShape) != 3:
            raise ValueError("Error: Input data should be 3D tensor with shape (batch_size, timesteps, input_dim).")

        if len(self.OutputShape) != 2:
            raise ValueError("Error: Target data should be 2D tensor with shape (batch_size, output_dim).")
        
        self.beta = beta
        
        #model build phase
        self.u_n = tf.placeholder(tf.float32, shape=[None] + list(self.InputShape[1:]), name='u')
        self.esn_cell = ESNCell(units=self.units, sr_scale=self.sr_scale, density=self.density, leaking_rate=self.leaking_rate)
        self.x_n = tf.keras.layers.RNN(self.esn_cell)(self.u_n)
        self.ones_tensor = tf.ones([tf.shape(self.x_n)[0],1], tf.float32)
        self.concatenated_x = tf.concat([self.x_n, self.ones_tensor], axis=1)
        self.W_out = tf.Variable(tf.random_normal(shape=[self.units+1,self.OutputShape[1]]))
        self.Y = tf.matmul(self.concatenated_x,self.W_out)
        
        #train phase
        self.sess.run(tf.global_variables_initializer())
        x_n_train = self.sess.run(self.concatenated_x, feed_dict={self.u_n: x})
        W_out_trained = Tikhonov_regularization(x_n_train, y, beta=self.beta)
        self.sess.run(self.W_out.assign(W_out_trained,read_value=False))
        self.trained = True
    
    def predict(self, x):
        """
        perform Regression on samples x.

        Parameters
        x : input data with shape (batch_size, timesteps, input_dim)
        
        Returns
        y_hat : predicted values for samples x(batch_size, output_dim)
        
        """
        InputShape_for_predict = x.shape
        if len(InputShape_for_predict) != 3:
            raise ValueError("Error: Input data should be 3D tensor with shape (batch_size, timesteps, input_dim).")
        if InputShape_for_predict[1:] != self.InputShape[1:]:
            raise ValueError("timesteps or input_dim of x shape(batch_size, timesteps, input_dim) should be the same as trained one.")
        if self.trained == False:
            raise ValueError("Error: Make the ESN model fit before")
            
        y_hat = self.sess.run(self.Y, feed_dict={self.u_n: x})
        return y_hat
    
    def MSE_Score(self, x, y):
        """
        calculate MSE according to the given labels and predict values.

        Parameters
        x : input data with shape (batch_size, timesteps, input_dim)
        y : target data with shape (batch_size, output_dim)
        
        Returns
        MSE : predicted values for samples x(batch_size, output_dim)
        
        """
        InputShape_for_score = x.shape
        OutputShape_for_score = y.shape
        if len(InputShape_for_score) != 3:
            raise ValueError("Error: Input data should be 3D tensor with shape (batch_size, timesteps, input_dim).")
        if InputShape_for_score[1:] != self.InputShape[1:]:
            raise ValueError("timesteps or input_dim of x shape(batch_size, timesteps, input_dim) should be the same as trained one.")
        if len(OutputShape_for_score) != 2:
            raise ValueError("Error: Target data should be 2D tensor with shape (batch_size, output_dim).")
        if InputShape_for_predict[1:] != self.OutputShape_for_score[1:]:
            raise ValueError("output_dim of y shape(batch_size, output_dim) should be the same as trained one.")
            
        if self.trained == False:
            raise ValueError("Error: Make the ESN model fit before")
        y_tf = tf.constant(y)
        MSE = tf.reduce_mean(tf.square(y_tf - self.Y))
        MSE_ = self.sess.run(MSE, feed_dict={self.u_n: x})        
        return MSE_
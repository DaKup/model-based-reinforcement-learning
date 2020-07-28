import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorboard



# # option1: initial state + sequence of actions => return next state(s)
# # inputs = [batch, timesteps, feature]

# f_s_rnn = tf.keras.Sequential([
#       keras.layers.GRU(32,return_sequences=True),
#       keras.layers.Dense(s.size)
# ])
# f_s_rnn.compile(optimizer='adam', loss='mse')

# # option2: every state + every action => return next state
# # state model (state, action) -> state

# f_s = keras.Sequential([
#     keras.layers.Dense(32, input_shape=np.vstack([s, a]).reshape(-1).shape),
#     keras.layers.Activation('relu'),
#     keras.layers.Dense(32),
#     keras.layers.Activation('relu'),
#     keras.layers.Dense(s.size)#,
#     # keras.layers.Activation('tanh')
# ])
# f_s.compile(optimizer='adam', loss='mse')

# f_r = keras.Sequential([
#     keras.layers.Dense(32, input_shape=np.vstack([s, a]).reshape(-1).shape),
#     keras.layers.Activation('relu'),
#     keras.layers.Dense(32),
#     keras.layers.Activation('relu'),
#     keras.layers.Dense(1)#,
#     # keras.layers.Activation('tanh')
# ])
# f_r.compile(optimizer='adam', loss='mse')

# pi = keras.Sequential([
#     keras.layers.Dense(32, input_shape=s.reshape(-1).shape),
#     keras.layers.Activation('relu'),
#     keras.layers.Dense(32),
#     keras.layers.Activation('relu'),
#     keras.layers.Dense(a.size)#,
#     # keras.layers.Activation('tanh')
# ])
# pi.compile(optimizer='adam', loss='mse')


# pi_explore = lambda state : np.random.randint(low=a_min, high=a_max, size=ndim)
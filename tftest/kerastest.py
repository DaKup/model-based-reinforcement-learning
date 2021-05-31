import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM, GRU, Input
from tensorflow.keras import layers



def main():

    num_samples = 1000
    num_features = 7
    X = np.random.rand(num_samples, 2, 2, 3, 5)
    y = np.random.rand(num_samples, 2, 2)

    layer = GRU(512)

    batch_size=30
    timesteps=20
    features=15
    out = layer(np.random.rand([batch_size, timesteps, features]))


    model = Sequential([
        keras.Input(shape=(None, num_features)),
        layers.GRU(512),
        layers.BatchNormalization()
    ])

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(12, input_dim=num_features, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=15, batch_size=10)
    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy))



if __name__ == "__main__":
    main()

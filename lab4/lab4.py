import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, SimpleRNN, Reshape

x = np.linspace(-5, 5, 1000)
y = np.linspace(0, 10, 1000)
z = (x - y) * np.sin(x + y)
inputs = np.column_stack((x, y))

plt.plot(x, z)
plt.title("z = (x - y) * sin(x + y)")
plt.savefig("function.png")
plt.show()


def modelTesting(model, name):
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(inputs, z, epochs=20, batch_size=100)
    z_pred = model.predict(inputs).ravel()
    error = np.mean(np.abs(z - z_pred)) / np.mean(np.abs(z)) * 100
    print(f"{name}: середня відносна помилка = {error:.2f} %")
    plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(name)
    plt.savefig(f"{name}_loss.png")
    plt.show()
    plt.plot(x, z, label='Actual')
    plt.plot(x, z_pred, label='Predicted')
    plt.legend()
    plt.title(name)
    plt.savefig(f"{name}.png")
    plt.show()


def feedforwardCreation(layers, neurons):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(2,)))
    for i in range(layers-1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, name='output'))
    return model


def cascadeforwardCreation(layers, neurons):
    inputLayer = Input(shape=(2,), name='input')
    current = Dense(neurons, activation='relu')(inputLayer)
    for i in range(layers-1):
        concatenatedLayer = Concatenate()([inputLayer, current])
        current = Dense(neurons, activation='relu')(concatenatedLayer)
    outputLayer = Dense(1, name='output')(current)
    model = Model(inputs=inputLayer, outputs=outputLayer)
    return model


def elmanCreation(layers, neurons):
    model = Sequential()
    model.add(Reshape((1, 2), input_shape=(2,), name='input_reshape'))
    model.add(SimpleRNN(neurons, return_sequences=True, activation='relu'))
    for i in range(layers-1):
        model.add(SimpleRNN(neurons, return_sequences=True, activation='relu'))
    model.add(Dense(1, name='output'))
    model.add(Reshape((1,), name='output_reshape'))
    return model


if __name__ == "__main__":
    f1 = feedforwardCreation(1, 10)
    f2 = feedforwardCreation(1, 20)
    c1 = cascadeforwardCreation(1, 20)
    c2 = cascadeforwardCreation(2, 10)
    e1 = elmanCreation(1, 15)
    e2 = elmanCreation(3, 5)
    modelTesting(f1, "feedforward_1x10")
    modelTesting(f2, "feedforward_1x20")
    modelTesting(c1, "cascade_1x20")
    modelTesting(c2, "cascade_2x10")
    modelTesting(e1, "elman_1x15")
    modelTesting(e2, "elman_3x5")

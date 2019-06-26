# Cats and Dogs detection

## CNN using Tensorflow 2 Beta

### Dataset: 
* https://www.kaggle.com/tongpython/cat-and-dog

### Model:
```python
tf.keras.Sequential( [
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation=tf.nn.leaky_relu, input_shape=(192,192,1)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=tf.nn.leaky_relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation=tf.nn.leaky_relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation=tf.nn.leaky_relu),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
])
```

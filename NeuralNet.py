import tensorflow as tf

(x1,y1), (x2,y2) = tf.keras.datasets.mnist.load_data()
x1 = x1.reshape(60000,28,28,1)
x2 = x2.reshape(10000,28,28,1)

x1=x1/100
x2=x2/100
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32,(5,5),activation = tf.nn.relu, input_shape = (28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(50,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))


model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.SparseCategoricalCrossentropy())
model.fit(x1,y1,epochs=6)
model.evaluate(x2,y2)
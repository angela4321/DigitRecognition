import tensorflow as tf

(x1,y1), (x2,y2) = tf.keras.datasets.mnist.load_data()


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(200,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))


model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.SparseCategoricalCrossentropy())
model.fit(x1,y1,epochs=8);
model.evaluate(x2,y2);
print("here")
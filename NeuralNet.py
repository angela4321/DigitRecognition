import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train = ImageDataGenerator(rescale=0.1).flow_from_directory("/flowers",(100,100),batch_size=100)



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(50,(6,6),activation = tf.nn.relu, input_shape=(100,100,3)))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(50,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(6,activation = tf.nn.softmax))


model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.CategoricalCrossentropy())
model.fit(train,epochs = 5, steps_per_epoch=5)

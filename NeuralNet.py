import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

train = ImageDataGenerator(rescale=0.1).flow_from_directory("/Users/angela/Downloads/flowers",(100,100),batch_size=100)
labels = train.class_indices.keys()
print(labels)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(50,(6,6),activation = tf.nn.relu, input_shape=(100,100,3)))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(50,(6,6),activation = tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(50,(6,6),activation = tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(200,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(5,activation = tf.nn.softmax))


model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.CategoricalCrossentropy())
model.fit(train,epochs = 10, steps_per_epoch=10)

#test images
path = " "
while len(path)>0:
    print("Type in a path to an image")
    path = input()
    im = image.load_img(path, target_size=(100,100))
    pred = model.predict(np.expand_dims(image.img_to_array(im),0))
    maxI = 0
    print(pred)
    for i in range(len(pred[0])):
        if pred[0][i]>pred[0][maxI]:
            maxI = i

    cur = 0
    for l in labels:
        if cur==maxI:
            print(l)
            break
        cur+=1


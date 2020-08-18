import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

generator = ImageDataGenerator(rescale=0.1,validation_split=0.2)
train = generator.flow_from_directory("/Users/angela/Downloads/flowers",(100,100),batch_size=100,subset="training")
val = generator.flow_from_directory("/Users/angela/Downloads/flowers",(100,100),batch_size=100,subset="validation")

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
model.fit(train,epochs = 10, steps_per_epoch=10,validation_data=val)

#test images
path = " "
while len(path)>0:
    print("Type in a path to an image")
    path = input()
    try:
        im = image.load_img(path, target_size=(100,100))
    except:
        print("Not a valid path")
        continue
    pred = model.predict(np.expand_dims(image.img_to_array(im),0))
    max = np.argmax(pred[0])

    cur = 0
    for l in labels:
        if cur==max:
            print(l)
            break
        cur+=1


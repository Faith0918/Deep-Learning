# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')

# if gpus:
#     try:
#         tf.config.experimental.set_visible_devices(gpus[0],'GPU')
#     except Runtime as e:
#         print(e)

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import models
# 1. Load Data
# 2. Define Keras model
# 3. Compile Keras model
# 4. Fit Keras Model
# 5. Evaluate Keras Model
# 6. Tie it all together
# 7. Make predictions


if __name__ == "__main__":
    


    #1. Load Data

    (train_images, train_labels),(test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000,28,28,1))
    train_images = train_images.astype('float32')/255

    test_images = test_images.reshape((10000,28,28,1))
    test_images = test_images.astype('float32')/255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    #2. Define Keras Model
    model = models.Sequential()
    # First Convolution Layer
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    # model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1), padding = 'same'))
    # model.add(layers.MaxPooling2D((2,2)))
    # Second Convolution Layer
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    # model.add(layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))
    # model.add(layers.MaxPooling2D(2,2))
    # Third Convolution Layer
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    # model.add(layers.Conv2D(64,(3,3),activation='relu', padding = 'same'))
    model.add(layers.Flatten())

    model.summary()

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))



    #3. Fit Keras Model
    model.compile(optimizer ='rmsprop', loss='categorical_crossentropy',metrics =['accuracy'])
    model.fit(train_images, train_labels,epochs=5,batch_size = 64)

    #4. Evaluate Keras Model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(test_acc)

    # 5. Evaluate Keras Model

    # 6. Tie it all together
    
    # 7. Make predictions
    pass
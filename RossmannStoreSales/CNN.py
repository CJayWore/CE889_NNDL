import tensorflow as tf
import keras
from keras import layers


cnn_x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
cnn_x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

cnn_x_train = cnn_x_train.astype('float32')/255.0
cnn_x_test = cnn_x_test.astype('float32')/255.0

# create CNN architecture 3 Convolution layers + a Flatten layers + Fully Connected layer
cnn = keras.models.Sequential()
cnn.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
cnn.add(keras.layers.MaxPooling2D((2,2)))
cnn.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
cnn.add(keras.layers.MaxPooling2D((2,2)))
cnn.add(keras.layers.Dropout(0.3))
# cnn.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
# cnn.add(tf.keras.layers.MaxPooling2D((2,2)))

cnn.add(keras.layers.Flatten())
cnn.add(keras.layers.Dense(10))
cnn.add(keras.layers.Softmax())

# loss function
cnn_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile model
cnn.compile(optimizer='adam', loss=cnn_loss_fn, metrics=['accuracy'])

cnn.fit(cnn_x_train, y_train, epochs=10, validation_data=(cnn_x_test, y_test))

test_prediction_image_num = 0
image = cnn_x_test[test_prediction_image_num]
plt.imshow(image, cmap='gray')

print("CNN Prediction: " + str(np.argmax(cnn.predict(tf.expand_dims(cnn_x_test[test_prediction_image_num], axis=0)))))

cnn.summary()

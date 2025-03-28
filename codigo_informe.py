# %%
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

# %%

# Dataset original
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=5,
    validation_split=0.2,
    width_shift_range=0.25,
    height_shift_range=0.25
)
test_datagen = datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
train_data = datagen.flow_from_directory(
    directory="./datasets/FER2013/train",
    color_mode="grayscale",     
    batch_size=64,
    target_size=(48,48),
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    directory="./datasets/FER2013/test",
    color_mode="grayscale",
    batch_size=64,
    target_size=(48,48),
    shuffle=False
)

# %%
# Red neuronal densa
# %%
model_informe_densa = models.Sequential()
model_informe_densa.add(Flatten())

model_informe_densa.add(layers.Dense(64, activation='relu'))
model_informe_densa.add(layers.Dense(128, activation='relu'))
model_informe_densa.add(Dense(7, activation='softmax'))

model_informe_densa.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
history_informe_densa = model_informe_densa.fit(train_data, epochs=15,
                    validation_data=test_data)

# %%
test_loss, test_acc = model_informe_densa.evaluate(test_data, verbose=2)

print(test_acc)
plt.plot(range(1,len(history_informe_densa.history['accuracy'])+1),history_informe_densa.history['accuracy'])
plt.plot(range(1,len(history_informe_densa.history['val_accuracy'])+1),history_informe_densa.history['val_accuracy'])
plt.title('History of Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['training','validation'])

# %%
# ---------------------------------------------------------------
# Red convolucional 2D de prueba.
model_informe_conv = models.Sequential()
model_informe_conv.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model_informe_conv.add(layers.MaxPooling2D((2, 2)))

model_informe_conv.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_informe_conv.add(layers.MaxPooling2D((2, 2)))

model_informe_conv.add(layers.Conv2D(32, (5, 5), activation='relu'))
model_informe_conv.add(layers.MaxPooling2D((2, 2)))

model_informe_conv.add(Flatten())
model_informe_conv.add(Dense(7, activation='softmax'))

# %%
model_informe_conv.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
history_informe_conv = model_informe_conv.fit(train_data, epochs=15,
                    validation_data=test_data)

# %%
test_loss, test_acc = model_informe_conv.evaluate(test_data, verbose=2)

print(test_acc)
plt.plot(range(1,len(history_informe_conv.history['accuracy'])+1),history_informe_conv.history['accuracy'])
plt.plot(range(1,len(history_informe_conv.history['val_accuracy'])+1),history_informe_conv.history['val_accuracy'])
plt.title('History of Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['training','validation'])
# %%
# ------------------------------------------------------------
# Modelo con mejor resultado obtenido

model_optimo = models.Sequential()
model_optimo.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), strides=1, padding='same'))
model_optimo.add(layers.BatchNormalization())
model_optimo.add(layers.MaxPooling2D((2, 2)))
model_optimo.add(layers.Dropout(0.25))

model_optimo.add(layers.Conv2D(128, (5, 5), activation='relu', input_shape=(48, 48, 1), strides=1, padding='same'))
model_optimo.add(layers.BatchNormalization())
model_optimo.add(layers.MaxPooling2D((2, 2)))
model_optimo.add(layers.Dropout(0.25))

model_optimo.add(layers.Conv2D(512, (3, 3), activation='relu', input_shape=(48, 48, 1), strides=1, padding='same'))
model_optimo.add(layers.BatchNormalization())
model_optimo.add(layers.MaxPooling2D((2, 2)))
model_optimo.add(layers.Dropout(0.25))

model_optimo.add(Flatten())

model_optimo.add(layers.Dense(256, activation='relu'))
model_optimo.add(layers.BatchNormalization())
model_optimo.add(layers.Dropout(0.5))

model_optimo.add(layers.Dense(512, activation='relu'))
model_optimo.add(layers.BatchNormalization())
model_optimo.add(layers.Dropout(0.5))

model_optimo.add(Dense(7, activation='softmax'))

# %%
model_optimo.summary()

# %%
auto_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=2,
    min_lr=0.00001,
    model='auto'
)
model_optimo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history_optimo = model_optimo.fit(train_data, epochs=50,
                    validation_data=test_data,
                    callbacks=[auto_lr])
# %%
test_loss, test_acc = model_optimo.evaluate(test_data, verbose=2)

# %%
print(test_acc)
plt.plot(range(1,len(history_optimo.history['accuracy'])+1),history_optimo.history['accuracy'])
plt.plot(range(1,len(history_optimo.history['val_accuracy'])+1),history_optimo.history['val_accuracy'])
plt.title('History of Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['training','validation'])

# -----------------------------------------------------
# -----------------------------------------------------

# %%

# Dataset balanceado
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=5,
    validation_split=0.2,
    width_shift_range=0.25,
    height_shift_range=0.25
)
test_datagen = datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
train_data = datagen.flow_from_directory(
    directory="./datasets/BalancedFER2013/train",
    color_mode="grayscale",     
    batch_size=64,
    target_size=(48,48),
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    directory="./datasets/BalancedFER2013/test",
    color_mode="grayscale",
    batch_size=64,
    target_size=(48,48),
    shuffle=False
)

# %%
# -----------------------------------------
# Las mismas redes que antes pero con una densa de 6 al final como salida

# Red neuronal densa
# %%
model_informe_densa = models.Sequential()
model_informe_densa.add(Flatten())

model_informe_densa.add(layers.Dense(64, activation='relu'))
model_informe_densa.add(layers.Dense(128, activation='relu'))
model_informe_densa.add(Dense(6, activation='softmax'))

model_informe_densa.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
history_informe_densa = model_informe_densa.fit(train_data, epochs=15,
                    validation_data=test_data)

# %%
test_loss, test_acc = model_informe_densa.evaluate(test_data, verbose=2)

print(test_acc)
plt.plot(range(1,len(history_informe_densa.history['accuracy'])+1),history_informe_densa.history['accuracy'])
plt.plot(range(1,len(history_informe_densa.history['val_accuracy'])+1),history_informe_densa.history['val_accuracy'])
plt.title('History of Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['training','validation'])

# %%
# ---------------------------------------------------------------
# Red convolucional 2D de prueba.
model_informe_conv = models.Sequential()
model_informe_conv.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model_informe_conv.add(layers.MaxPooling2D((2, 2)))

model_informe_conv.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_informe_conv.add(layers.MaxPooling2D((2, 2)))

model_informe_conv.add(layers.Conv2D(32, (5, 5), activation='relu'))
model_informe_conv.add(layers.MaxPooling2D((2, 2)))

model_informe_conv.add(Flatten())
model_informe_conv.add(Dense(6, activation='softmax'))

# %%
model_informe_conv.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
history_informe_conv = model_informe_conv.fit(train_data, epochs=15,
                    validation_data=test_data)

# %%
test_loss, test_acc = model_informe_conv.evaluate(test_data, verbose=2)

print(test_acc)
plt.plot(range(1,len(history_informe_conv.history['accuracy'])+1),history_informe_conv.history['accuracy'])
plt.plot(range(1,len(history_informe_conv.history['val_accuracy'])+1),history_informe_conv.history['val_accuracy'])
plt.title('History of Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['training','validation'])
# %%
# ------------------------------------------------------------
# Modelo con mejor resultado obtenido

model_optimo = models.Sequential()
model_optimo.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), strides=1, padding='same'))
model_optimo.add(layers.BatchNormalization())
model_optimo.add(layers.MaxPooling2D((2, 2)))
model_optimo.add(layers.Dropout(0.25))

model_optimo.add(layers.Conv2D(128, (5, 5), activation='relu', input_shape=(48, 48, 1), strides=1, padding='same'))
model_optimo.add(layers.BatchNormalization())
model_optimo.add(layers.MaxPooling2D((2, 2)))
model_optimo.add(layers.Dropout(0.25))

model_optimo.add(layers.Conv2D(512, (3, 3), activation='relu', input_shape=(48, 48, 1), strides=1, padding='same'))
model_optimo.add(layers.BatchNormalization())
model_optimo.add(layers.MaxPooling2D((2, 2)))
model_optimo.add(layers.Dropout(0.25))

model_optimo.add(Flatten())

model_optimo.add(layers.Dense(256, activation='relu'))
model_optimo.add(layers.BatchNormalization())
model_optimo.add(layers.Dropout(0.5))

model_optimo.add(layers.Dense(512, activation='relu'))
model_optimo.add(layers.BatchNormalization())
model_optimo.add(layers.Dropout(0.5))

model_optimo.add(Dense(6, activation='softmax'))

# %%
model_optimo.summary()

# %%
auto_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=2,
    min_lr=0.00001,
    model='auto'
)
model_optimo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history_optimo = model_optimo.fit(train_data, epochs=50,
                    validation_data=test_data,
                    callbacks=[auto_lr])
# %%
test_loss, test_acc = model_optimo.evaluate(test_data, verbose=2)

# %%
print(test_acc)
plt.plot(range(1,len(history_optimo.history['accuracy'])+1),history_optimo.history['accuracy'])
plt.plot(range(1,len(history_optimo.history['val_accuracy'])+1),history_optimo.history['val_accuracy'])
plt.title('History of Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['training','validation'])
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout

# Step-1
# Initialising the CNN
classifier = Sequential()

# Step-2
# Adding first convolutional layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding third convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step-3
# Flattening
classifier.add(Flatten())

# Step-4 
# Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 7, activation = 'softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.5,
                                   horizontal_flip = False)

img_set = datagen.flow_from_directory('faces', target_size = (64, 64), 
                                      batch_size = 15, class_mode = 'categorical')

classifier.fit_generator(img_set,
                         samples_per_epoch = 37219,
                         nb_epoch = 100)

classifier.save('model.h5')
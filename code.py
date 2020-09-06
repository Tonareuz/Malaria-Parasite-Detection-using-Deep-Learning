# Part 1 - Building the CNN
#Importing the libraries for reading the images(SciKit)
from skimage.io import imread
from skimage.transform import resize

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(filters = 32, kernel_size = (3, 3), input_shape = (128, 128, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding a third convolutional layer 
classifier.add(Convolution2D(filters = 128, kernel_size = (3,3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding a fourth convolutional layer 
classifier.add(Convolution2D(filters = 256, kernel_size = (3,3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection(adding hidden layers for the classic neural network)
classifier.add(Dense(units = 128,activation="relu"))
classifier.add(Dense(units = 128,activation="relu"))
classifier.add(Dense(units = 1,activation="sigmoid"))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

#Compiling the classifier using training and test set
classifier.fit_generator(training_set,
                         steps_per_epoch = 250,   #steps_per_epoch = samples_per_epoch/batch_size
                         epochs = 25,             #time taken for one forward and backward pass on a set of samples 
                         validation_data = test_set,
                         validation_steps = (2000/32))

class_labels = {v: k for k, v in training_set.class_indices.items()}
 
#the file we want to evaluate for malaria(destination of the image)

img = imread(path_to_file) 
img = resize(img,(64,64))
img = np.expand_dims(img,axis=0)
 
if(np.max(img)>1):
    img = img/255.0
 
prediction = classifier.predict_classes(img) 
 
print(class_labels[prediction[0][0]])

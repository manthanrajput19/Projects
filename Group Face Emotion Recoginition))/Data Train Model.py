from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

train_data_dir = 'Data/train'#train path to take data  
validation_data_dir = 'Data/test'#test patha to take data 
#train_datagen ke ander Image data generator function jyga jo python ki predifined class h 
#imagedatagenerator ek data argument(parameter) create krega during traning
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')
#validate the Image data and rescale is a parameter of this clss(no arguments) 
validation_datagen = ImageDataGenerator(rescale=1./255)
#flow from directory is a method jo train_datagen se image train_generator mai bejega
# Generate krega batches of augmented data for training
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)
#Generate krega batches of augmented data for validation
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)
#create some labels to identify konsa emotion show hora h
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# Get an example batch of images and labels from the training generator
img, label = train_generator.__next__()
# Define the model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))
# Compile krega model ko 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
train_path = "Data/train/"
test_path = "Data/test"
# Count the number of training and test images
num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)
print("Number of training images:", num_train_imgs)
print("Number of test images:", num_test_imgs)
# Train krege model ko jisse wo validate data or train data ko laker ek HDF5 file stands for Hierarchical Data Format 5 file generate krega  
epochs = 100
history = model.fit(
    train_generator,
    steps_per_epoch=num_train_imgs // 170,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_test_imgs // 170
)
#.save is use to create a file which  name is model_file_100epochs.h5
model.save('facialemotionmodel.h5')
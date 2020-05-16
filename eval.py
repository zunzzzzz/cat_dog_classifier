import os
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

checkpoint = 'checkpoint.h5'

base_dir = 'cats_and_dogs_small' # Would be absolute path to dataset when scoring.

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation') 
# test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# test_cats_dir = os.path.join(test_dir, 'cats')
# test_dogs_dir = os.path.join(test_dir, 'dogs')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
            # Would be test set.
            # test_dir,
            validation_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

model = models.load_model(checkpoint)

score = model.evaluate(test_generator)

print('test loss: %f, test acc: %f' % (score[0], score[1]))











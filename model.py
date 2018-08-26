import csv
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten,  Dense, Lambda,  Cropping2D

def generator(samples, batch_size=32):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        #shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #name = './IMG/'+batch_sample[0].split('/')[-1]
                angle = float(batch_sample[3])
                name= batch_sample[0]
                camera = np.random.choice(['center', 'left', 'right', 'flip'])
                #print(camera)
                
                if camera == 'left':
                    angle += 0.20
                    name = batch_sample[1]
                elif camera == 'right':
                    angle -= 0.20
                    name= batch_sample[2]
                elif camera == 'flip' :   
                    angle = -1*angle
                    
                image = cv2.imread(name)
                if camera == 'flip' :
                    image = cv2.flip(image,1)
                
                #print(camera, name)
                images.append(image)
                angles.append(angle)

            
            # trim image to only see section with road
            X_batch = np.array(images)
            y_batch = np.array(angles)
            yield sklearn.utils.shuffle(X_batch, y_batch)
            
def get_model():
    model = Sequential()

    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))

    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
              
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Flatten())
    # Finally a single output, since this is a regression problem
    model.add(Dense(100))
    model.add(Dense(50))  
    model.add(Dense(10))  
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model            


if __name__ == "__main__":
    BATCH_SIZE = 32

    lines = []
    with open('./driving_log.csv') as csvfile:
       reader = csv.reader(csvfile)
       for line in reader:
          lines.append(line)
        
    print(len(lines))      

    used_lines = lines[0:1800] + lines[9200:]
    train_samples, validation_samples = train_test_split(used_lines, test_size=0.2)


    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    model = get_model()

    model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), nb_epoch=3)

    print("Saving model file.")

    model.save('model.h5')  # always save model after training 
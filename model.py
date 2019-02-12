import csv
import cv2
import numpy as np

    
def load_line(line, img_idx, angle_correction=0.):
    '''
    Load image and the angle data of each line in driving_log.csv
    '''
    current_path = './data/IMG/' + line[img_idx].split('/')[-1]
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #measurement * 1.2 due to left turn not enough in simulation model. 
    # And maybe it was caused by flip center images.
    measurement = float(line[3])*1.2 + angle_correction
    return image, measurement

def load_data_4(lines):
    '''
    Load image data of each line. Include 4 type images, they are center image, flip center images, left images and right images.
    The angle of left image should + 0.2 for correction and angle of right should -0.2 for correction.
    '''
    images = []
    measurements = []
    count = 0
    for line in lines:
        # read center image
        image, measurement = load_line(line, img_idx=0)
        images.append(image)
        measurements.append(measurement)
        # flip center image
        images.append(cv2.flip(image,1))
        measurements.append(measurement*-1.0)
        # read left
        image, measurement = load_line(line, 1, 0.2)
        images.append(image)
        measurements.append(measurement)
        # read right
        image, measurement = load_line(line, 1, -0.2)
        images.append(image)
        measurements.append(measurement)
        
        # processing count
        count += 4
        if (count%500 == 0):
            print('. ', end='', flush=True)
    # Return data
    print("Load data done! Total counts=", count)
    X = np.array(images)
    y = np.array(measurements)
    return X, y

def load_data_3(lines):
    '''
    Load images data of each line. Include 3 type images, they are center image, left images and right images. 
    The angle of left images will add +0.2 for correction and angle of right add -0.2 correction.
    '''
    images = []
    measurements = []
    count = 0
    for line in lines:
        # read center image
        image, measurement = load_line(line, img_idx=0)
        images.append(image)
        measurements.append(measurement)
        # read left
        image, measurement = load_line(line, 1, 0.2)
        images.append(image)
        measurements.append(measurement)
        # read right
        image, measurement = load_line(line, 1, -0.2)
        images.append(image)
        measurements.append(measurement)
        
        # processing count
        count += 3
        if (count%500 == 0):
            print('. ', end='', flush=True)
    # Return data
    print("Load data done! Total counts=", count)
    X = np.array(images)
    y = np.array(measurements)
    return X, y

def load_data(lines):
    '''
    Load images data of each line. Include 2 type images, they are center image, fliped center images.  
    '''
    images = []
    measurements = []
    count = 0
    for line in lines:
        source_path = line[0]
        center_filename = source_path.split('/')[-1]
        #print(center_filename)
        current_path = './data/IMG/' + center_filename
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        images.append(cv2.flip(image,1))
        count += 2
        if (count%500 == 0):
            print('. ', end='', flush=True)

        # get the measurement
        measurement = float(line[3])*1.5
        measurements.append(measurement)
        measurements.append(measurement*-1.0)

    print("Loading Data done!")
    print(len(images), images[0].shape)

    X = np.array(images)
    y = np.array(measurements)
    return X, y

# Prepare images or X(training) samples and y(flag) samples.
lines = []
with open('./data/driving_log.csv') as cvsfile:
    csv_handler = csv.reader(cvsfile)
    for line in csv_handler:
        lines.append(line)
        
#X_samples, y_samples = load_data_4(lines)
#X_samples, y_samples = load_data_3(lines)
X_samples, y_samples = load_data(lines)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D


def LeNet(model, X_train, y_train):
    '''
    Pure LeNet
    '''
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape = (160,320, 3)))
    model.add(Lambda(lambda x:(x/255.0)-0.5))
    model.add(Conv2D(6,(5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(6,(5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer = 'adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4)
    print('save model-LeNet.h5')
    model.save('model-LeNet.h5')

def LeNet_modified(model, X_train, y_train):
    '''
    Modified LeNet which pass traffic sign classification task
    Convlution(61x316x6) with relu
    max_pooling(30x158x6)
    dropout(0.2)
    Convolution(28x156x32) with relu
    max_pooling(14x78x32)
    flatten(34944*3)
    dense(384)
    dropout(0.5)
    dense(84)
    dense(1)
    '''
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape = (160,320, 3))) #65x320x3
    model.add(Lambda(lambda x:(x/255.0)-0.5))
    model.add(Conv2D(6,(5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(32,(3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(384))
    model.add(Dropout(0.5))
    model.add(Dense(84))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer = 'adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)
    model.save('model-LeNet-modified.h5')
    print('save model-LeNet-modified.h5')
    
def CNN_1(model, X_train, y_train):
    '''
    CNN modification 1.
    Convolution(86x316x24) with relu
    max_poolling(43x158x24)
    dropout(0.3)
    Convolution(39x154x32) with relu
    max_pooling(19x77x32)
    flat(46816*3)
    Dense(32)
    dropout(0.25)
    Dense(16)
    Dense(1)
    '''
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape = (160,320, 3))) #90x320x3
    model.add(Lambda(lambda x:(x/255.0)-0.5))
    model.add(Conv2D(24,(5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))
    model.add(Conv2D(32,(5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dropout(0.25))
    model.add(Dense(16))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer = 'adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=20)
    model.save('model2.h5')
    
def CNN_2(model, X_train, y_train):
    '''
    CNN net work without dropout.
    Convolution(30x158x24) with relu
    Convolution(13x77x36) with relu
    Convolution(5x37x48) with relu
    Convolution(3x17x64) with relu
    Convolution(1x7x64) with relu
    flatten(448x3)
    dense(100)
    dense(50)
    dense(10)
    dense(1)
    '''
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape = (160,320, 3))) # 65x320x3
    model.add(Lambda(lambda x:(x/255.0)-0.5))
    model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu')) #, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer = 'adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)
    model.save('model.h5')
    
print('Configuration model')
model = Sequential()
#LeNet(model, X_samples, y_samples)
#LeNet_modified(model, X_samples, y_samples)
#CNN_1(model, X_samples, y_samples)
CNN_2(model, X_samples, y_samples)




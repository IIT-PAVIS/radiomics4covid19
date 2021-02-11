# Library Versions
# The following code was developed using the following libraries  
# tensorflow = 1.4
# keras = 2.1.5
# sklearn = 0.22.2
# matplotlib = 3.0.3


import os
from sklearn.metrics import confusion_matrix
import pandas as pd
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import cv2
from sklearn.utils import shuffle
from keras.utils import np_utils
import csv
from keras.layers import *
from keras import *
import pathlib
from keras.applications.resnet50 import ResNet50


# Setting GPU ID
os.environ["CUDA_VISIBLE_DEVICES"]="0"

BasePath = os.getcwd()
print("BASE PATH : ",BasePath)

#Parameters
num_class = 2
epochs = 50
batch_size = 32
ImgSize = 224
learn = 0.0005
Experiment_Name = 'ResNet50-10-Fold-Only_Images'
# Data Files
train_data_path = BasePath + '/Data/data_files/Train.csv'
Validate_data_path = BasePath + '/Data/data_files/Val.csv'
test_data_path = BasePath + '/Data/data_files/Test.csv'


#Clinical Information Model
def create_mlp(dim):
	# define our MLP network
    model = Sequential()
    model.add(Dense(32, input_dim=dim, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(12, activation="relu"))
    return model

# Costum Data Generator for the multi Input
def load_samples(csv_file):
    print()
    data = pd.read_csv(csv_file)
    data = data[['FileName', 'Label', 'ClassName']]
    file_names = list(data.iloc[:,0])
    # Get the labels present in the second column
    labels = list(data.iloc[:,1])
    samples=[]
    for samp,lab in zip(file_names,labels):
        samples.append([samp,lab])
    return samples
def shuffle_data(data):
    data = shuffle(data)#,random_state=2)
    return data
def preprocessing(img,label):
    img = cv2.resize(img,(ImgSize,ImgSize))
    img = img/255
    label = np_utils.to_categorical(label, num_class)
    return img,label
def normalize(lst):
    s = sum(lst)
    return (lambda x: float(x)/s, lst)
def st_t_onumber(x):
    import numbers
    # if any number
    if isinstance(x,numbers.Number):
        return x
    # if non a number try convert string to float or it
    for type_ in (int, float):
        try:
            return type_(x)
        except ValueError:
            continue
def data_generator(samples, batch_size, shuffle_data=True, resize=224):
    data_path = 'data_files/Train.csv'
    num_samples = len(samples)
    print("THE LENGTH = ", num_samples)
    while True:  # Loop forever so the generator never terminates
        #samples = shuffle(samples)
        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset + batch_size]
            # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []
            z_train = []
            # For each example
            for batch_sample in batch_samples:
                # Load image (X) and label (y)
                img_name = batch_sample[0]
                label = batch_sample[1]
                img = cv2.imread(os.path.join(img_name))
                img, label = preprocessing(img, label)
                # Add example to arrays
                X_train.append(img)
                y_train.append(label)
                sp = img_name.split("/")
                imgNameF = sp[2][:-4]
                with open(BasePath + '/Data/Clinical_Info.csv') as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    line_count = 0
                    for row in csv_reader:
                        if row[2] == imgNameF:
                            alpha = row[3:37]
                    z_train.append(alpha)
            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            Z_train = np.array(z_train)
            Z_train = np.asarray(Z_train, dtype=np.float32)
            yield [X_train, Z_train], y_train

# Model Training
def Train_Model(files):

    img_shape = (ImgSize, ImgSize, 3)
    inputs = Input(img_shape)

    xception = ResNet50(include_top=False, weights=None, input_shape=img_shape)
    xception.trainable = True
    outputs1 = xception(inputs)
    
    outputs = BatchNormalization()(outputs1)
    outputs = GlobalAveragePooling2D()(outputs)
    outputs = Dropout(0.25)(outputs)
    outputs = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(outputs)
    outputs2 = Dense(num_class, activation='softmax')(outputs)
    
    model = Model(inputs=[inputs], outputs=[outputs2])
    sgd = optimizers.SGD(lr=learn, nesterov=True)
    model.compile(optimizer=sgd, loss='hinge', metrics=['accuracy'])
    model.summary()

    #model.load_weights("/home/mohamad/Desktop/CDI/Vgg/Weights-ReNet50-Pneumonia3.h5")

    mlp = create_mlp(34)
    combinedInput = concatenate([outputs, mlp.output])

    # ***************************************************************************************
    # Select The appropiate model
    # Combined Model
    # x = Dense(num_classes, activation='softmax', kernel_initializer=initializer, kernel_regularizer=regularizers.l2())(combinedInput)
    # Only Clinical Info
    # x = Dense(num_class)(mlp.output)  # Clinical Info Only
    # Only Images
    x = Dense(num_class, activation='softmax')(outputs)  # Only Images
    # ***************************************************************************************

    model = Model(inputs=[inputs, mlp.input], outputs=x)
    sgd = optimizers.SGD(lr=learn, nesterov=True)
    model.compile(loss='hinge',
                  optimizer=sgd,
                  metrics=['accuracy'])


    # Plotting the Model Architecture
    SaveImgPath = BasePath+ "/Saved_Images"
    file_Image = pathlib.Path(SaveImgPath)
    if file_Image.exists():
        print("Saved Images Path : ", SaveImgPath)
    else:
        os.makedirs(SaveImgPath)
        print("Saved Images Path : ", SaveImgPath)
    plot_model(model, SaveImgPath + "/" + Experiment_Name + "_model.png", show_shapes=True)


    # Save Path for Wieghts Checkpoints
    SaveDirPath = BasePath+ "/Saved_Model"
    file_weights = pathlib.Path(SaveDirPath)
    if file_weights.exists():
        print("Weights Path : ",SaveDirPath)
    else:
        os.makedirs(SaveDirPath)
        print("Weights Path : ",SaveDirPath)
    Weights_path = SaveDirPath + '/CheckPointModel.h5'
    checkpoint = ModelCheckpoint(Weights_path, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]



    train_samples = load_samples(train_data_path)
    Val_samples = load_samples(Validate_data_path)
    num_train_samples = len(train_samples)
    num_Val_samples = len(Val_samples)
    print('number of train samples: ', num_train_samples)
    print('number of Validation samples: ', num_Val_samples)
    # Create generator
    train_generatorCustom = data_generator(train_samples, batch_size=batch_size)
    validation_generatorCustom = data_generator(Val_samples, batch_size=batch_size)
    STEP_SIZE_TRAIN = num_train_samples / batch_size
    STEP_SIZE_VALID = num_Val_samples / batch_size
    # Train the Model
    history = model.fit_generator(train_generatorCustom,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  epochs=epochs, validation_data=validation_generatorCustom,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=callbacks_list)
    # model.save_weights(BasePath + '/Saved_Model/'+ name +'_model.h5')
    print("Training Complete and Weights are saved")
    # Saving the Model
    model.save(SaveDirPath + "/Complete_Model_Weights.h5")


    # Plotting the Learning Curves
    # Accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    plt.savefig(SaveImgPath + '/Model_Accuracy_' + Experiment_Name + '.png')
    # Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    plt.savefig(SaveImgPath + '/Model_Loss_' + Experiment_Name + '.png')



    # ***************************************************
    # Testing the Saved Model
    # ***************************************************
    test_samples = load_samples(test_data_path)
    num_test_samples = len(test_samples)
    print('number of Test samples: ', num_test_samples)
    STEP_SIZE_Test = num_test_samples / batch_size
    Test_generatorCustom = data_generator(test_samples, batch_size=batch_size)
    model.load_weights(Weights_path)

    # Confution Matrix and Classification Report

    Y_pred = model.predict_generator(Test_generatorCustom, steps=STEP_SIZE_Test)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    # X_Val, y_val = next(iter(validation_generatorCustom))
    df = pd.read_csv(test_data_path, usecols=['Label'])  # Reading Ground Truths for comparision with prediction's
    y_true = sum(df.values.tolist(), [])

    array = confusion_matrix(y_true, y_pred)
    print(Experiment_Name)
    print(array)
    TP = array[0][0]
    FP = array[0][1]
    FN = array[1][0]
    TN = array[1][1]
    # print(array)
    print(" TP : " + TP.__str__() + ", TN : " + TN.__str__() + ", FN : " + FN.__str__() + ", FP : " + FP.__str__())
    # Saving the Result in a Text File

    ResultPath = BasePath + "/Results"
    file_Result = pathlib.Path(ResultPath)
    if file_Result.exists():
        print("Result Path : ", ResultPath)
    else:
        os.makedirs(ResultPath)
        print("Result Path : ", ResultPath)
    f = open( ResultPath + "/" +Experiment_Name, "a" )
    f.write(" TP " + TP.__str__() + " TN " + TN.__str__() + " FN " + FN.__str__() + " FP " + FP.__str__())
    f.close()
    # print("Accuracy : ", accuracy_score(y_true, y_pred))
    percision = TP / (TP + FP)
    recall = TP / (TP + FN)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("Percision : ", round(percision, 4))
    print("Recall : ", round(recall, 4))
    print("Accuracy : ", round(Accuracy, 4))


if __name__=='__main__':
    Data_file = BasePath + "/Data/data_files"
    Train_Model(Data_file)

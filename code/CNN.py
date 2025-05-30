from data_processing import *
from tensorflow.keras import models, layers
import tensorflow_datasets as tfds
import json
import os

if __name__ == "__main__":
    #print(tf.__version__)
    #print(tf.config.list_physical_devices())
    
    Custom = True
    
    ### CUSTOM DATASET (DEEPFASHION2) ###
    if Custom:
        train_ds, val_ds = ImportDataset("keras_dataset_5000",batch_size=32) # The batch_size overrides the fit function parameter
        CheckData(train_ds)
        #Normalize(train_ds,val_ds) # Added to the model
        #CheckNaN(val_ds)
        #CheckNaN(train_ds)
        Perf(train_ds, val_ds) # Memory usage optimizer

        output = len(class_names)
        input_shape  = (32, 32, 3) # Defined in ImportDataset function

    ### TENSORFLOW DATASET ###

    else:
        tfds.enable_progress_bar()
        #print(tfds.list_builders())
        (train_ds, val_ds), info = tfds.load(name='fashion_mnist',
                                    data_dir="./fashion_mnist", # Destination dir
                                    shuffle_files=True, # Shuffles the input files
                                    with_info=True, # Returns info about the dataset
                                    as_supervised=True,  # Return (image, label) tuples
                                    # Divide the predefined training set according to the specified splits
                                    split=['train[:17%]','train[17%:24%]'], 
                                    batch_size=32 # this overrides the fit function parameter 
                                    ) 
                                    
        # Data info
        num_classes = info.features['label'].num_classes
        print(num_classes)
        image_shape = info.features.shape['image']
        print(image_shape)
        print(info.features.dtype)

        # Visualize image
        #get_label_name = info.features['label'].int2str
        #
        #for image, label in train_ds.take(1):
        #    plt.imshow(np.squeeze(image[0]).astype('uint8'))
        #    plt.title(get_label_name(label[0]))
        #    plt.show()
        
        output = num_classes
        input_shape = image_shape

        Perf(train_ds,val_ds)

    ### CREATE MODEL ###

    ## Convolutional + Pooling layers
    model = models.Sequential()
    
    #model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    #model.add(layers.MaxPooling2D((2, 2))) # strides equal to pool_size if not specified
    
    # Normalization + Centering layer
    model.add(layers.Rescaling(scale=1./127.5,offset=-1,input_shape=input_shape)) 
    # Padding='same' keeps the output size of a channel equal to the input size
    model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))) # strides equal to pool_size if not specified

    model.add(layers.Conv2D(filters=64, kernel_size=(5,5), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=64, kernel_size=(5,5), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=64, kernel_size=(5,5), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.summary()

    # Assuming that the input shape is (32,32,3) and 32 (5x5) filters are applied, the output shape 
    # of the convolutional layer is (32,32,32) (with padding to keep the original image size). 
    # This may seem strange since the input channels are 3 so the output size should be (32,32,32*3).
    # However Keras executes the convolution with kernels of size (5,5,3) so each step produces a single
    # monochrome output value.
    # Ref: https://stackoverflow.com/questions/55444120/understanding-the-output-shape-of-conv2d-layer-in-keras 

    ## Fully connected layers + Dropout + Softmax
    model.add(layers.Flatten())

    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output, activation='softmax'))

    #model = tf.keras.applications.vgg16.VGG16( # Doesn't start due to insufficient memory 
    #    include_top=False,
    #    weights=None,
    #    input_tensor=None,
    #    input_shape=input_shape,
    #    pooling='max',
    #    classes=10,
    #    classifier_activation='softmax'
    #)

    model.summary()    

    ## Model compilation
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    # Create callback to save checkpoints during training
    #checkpoint_path = "./training/cp-{epoch:04d}-{accuracy:.2f}-{val_accuracy:.2f}.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    #
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                save_weights_only=True,
    #                                                verbose=1)
    
    ## Training and Validation
    # Jetson Nano resources
    # - Global memory: 3964 MBytes (4156432384 bytes)
    # IMPORTANT: Jetson’s is a shared memory system, which indicates that the physical memory can be used via CPU or GPU
    # - L2 cache size: 262144 bytes
    epochs = 10
    training_batch = 32
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=epochs,
                        #callbacks=[cp_callback],
                        batch_size=training_batch)
    
    # Get results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    # Record results
    fp = open("./training_results.json",'w')
    res = {"acc":acc, "val_acc":val_acc, "loss":loss, "val_loss":val_loss, 'epochs':epochs} 
    json.dump(res,fp)
    fp.close()

    #plt.figure(figsize=(8, 8))
    #plt.subplot(1, 2, 1)
    #plt.plot(epochs_range, acc, label='Training Accuracy')
    #plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    #plt.legend(loc='lower right')
    #plt.title('Training and Validation Accuracy')

    #plt.subplot(1, 2, 2)
    #plt.plot(epochs_range, loss, label='Training Loss')
    #plt.plot(epochs_range, val_loss, label='Validation Loss')
    #plt.legend(loc='upper right')
    #plt.title('Training and Validation Loss')
    #plt.show()

    
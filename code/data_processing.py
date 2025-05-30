import pathlib
import tensorflow as tf
from constants import class_names
import numpy as np
import matplotlib.pyplot as plt

def ImportDataset(src_path:str,batch_size:int=1,dst_path:str=None,save:bool=False,compr:bool=False):
    """ Import a dataset as a tf.data.Dataset\n
    Parameters:
        src_path: root of the dataset
        batch_size: images that make a single sample in the dataset once imported. For example batch_size=32
        will create a dataset with samples made of 32 images.
        WARNING: the batch size for training still has to be specified  
        dst_path: where to save the dataset
        save: save the dataset in a lightweight format. Use the LoadDataset(...) function to retrieve it\n
        compr: 'True' compress the dataset when saving. The same option has to be specified when loading\n
    Returns:
        train_ds: training dataset
        val_ds: validation dataset\n
    Outputs:
        Create the train and validation directories where the dataset is saved\n"""
    
    ## Total images
    data_dir = pathlib.Path(src_path)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(f"TOTAL IMAGES: {image_count}")

    ## Load database as tf.data.Database

    # Data are batched as 1 32x32 image
    img_height = 32
    img_width = 32
    batch_size = batch_size  # None avoids batching --> I think it considers every pixel as a data point

    # Shuffle=True by default so each iteration with for loop, iterator, etc will
    # give a different data order
    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    class_names=class_names)
    # Labels are inferred from dir names but the class names is specified
    # to force the labels order, otherwise it is alphabetical

    #print(train_ds.class_names)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    class_names=class_names)

    #print(val_ds.class_names)

    ## Save dataset as tf.data.Database

    #if save:
    #    if compr:
    #        compr = 'GZIP'
    #    else:
    #        compr = None
    #    train_ds.save(dst_path+"/train_TF", compression=compr)
    #    val_ds.save(dst_path+"/val_TF", compression=compr)
    #
    return train_ds, val_ds

def LoadDataset(src_path:str,compr:bool=False):
    """ Load the previously saved dataset\n
    Parameters:
        src_path: path the dataset saved to
        compr: specify whether tha dataset was compressed upon saving\n
    Returns:
        train_ds: training dataset
        val_ds: validation dataset\n"""
    if compr:
        compr = 'GZIP'
    else:
        compr = None
    train_ds = tf.data.Dataset.load(src_path+"/train_TF",compression=compr)
    val_ds = tf.data.Dataset.load(src_path+"/val_TF",compression=compr)
    return train_ds, val_ds

def CheckData(ds:tf.data.Dataset):
    """ Check the some of the passed dataset data\n
    Parameters:
        ds: dataset\n
    WARNING: if the imported dataset batch_size is not 1 this function may throw some errors\n"""
    ## Check data
    for image_batch, labels_batch in ds:
        print(f"IMAGE SHAPE: {image_batch.shape}")
        print(f"LABEL SHAPE: {labels_batch.shape}")
        break

    #images = []
    #labels = []
    ## Collect 9 images from the dataset
    #for img, label in ds.take(9):
    #    images.append(img)
    #    labels.append(label)

    #plt.figure(figsize=(8, 8))
    #for i in range(9):
    #    #print(images[i])
    #    #print(labels[i])
    #    ax = plt.subplot(3, 3, i + 1)
    #    plt.imshow(np.squeeze(images[i]).astype("uint8"))
    #    plt.title(class_names[labels[i][0]])
    #    plt.axis("off")
    #plt.show()
    # 32x32 seems very off but that's what the paper use

def Normalize(train_ds:tf.data.Dataset,val_ds:tf.data.Dataset):
    """ Normalize training and validation datasets\n
    Parameters:
        train_ds: training dataset 
        val_ds: validation dataset\n"""
    ## Rescale layer to normalize pixels value
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    # Apply the rescaling layer to the dataset (or add it to the model)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    image_batch, labels_batch = next(iter(train_ds))
    first_image = image_batch[0]
    #print(first_image)
    # Notice the pixel values are now in `[0,1]`.
    print(f"Max after normalization: {np.min(first_image)}")
    print(f"Max after normalization: {np.max(first_image)}")

def CheckNaN(ds:tf.data.Dataset):
    """ Check for NaN values in the dataset\n
    Parameters:
        ds: the dataset to be checked\n"""
    ## Check for NaN values
    for img,label in ds:
      #print(img)
      #print(img.numpy())
      #print(label.numpy())
      is_nan_img = np.any(np.isnan(img.numpy()))
      is_nan_lbl = np.any(np.isnan(label.numpy()))
      if (is_nan_img == True or is_nan_lbl == True):
        print("NaN value detected")
        print(label)
        print(img)

    # NO NaN in both train_ds and val_ds --> Not the cause of loss function divergence

def Perf(train_ds:tf.data.Dataset,val_ds:tf.data.Dataset):
    """ Improve performances by optimizing the memory handling\n
    Check Tensorflow documentation for more info\n
    Parameters:
        train_ds: training dataset 
        val_ds: validation dataset\n"""
    ## Dataset handling performance improvement
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

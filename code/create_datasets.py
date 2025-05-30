import os
import json
import shutil
from math import ceil
from constants import class_names
from PIL import Image

def ClothingFinder(oneItemOnly:bool=False):
    """ Organize the images in the Deepfashion2 dataset starting from the
        training and validation sets json annotations.\n
        Requires the following directory tree to work properly:
            |dataset\n
            \t|train\n
            \t\t|annos\n
            \t\t|image\n
            \t|validation\n
            \t\t|annos\n
            \t\t|image\n
        'train' and 'validation' are the extracted Deepfashion2 archives\n
        Parameters:
            oneItemOnly: if True only images with 1 item are recorded. This is useful
            if you do not want to crop the images when creating the dataset with the CreateKerasCompatibleDS method
            but you cannot classify multiple objects in the same image\n 
        Outputs in the execution directory:
            3 files describing the images distribution\n"""
    
    if oneItemOnly:
        oneItem = "_oneItem"
    else:
        oneItem = ""

    class_stats = {}
    for i in range(len(class_names)):
        class_stats[i] = {"name": class_names[i], "num": 0}

    for phase in ("validation", "train"):
        print("### Processing {} set ###".format(phase))
        dir = f"./dataset/{phase}/annos/"
        #print(f"Files in the directory: {dir}")

        files = os.listdir(dir)
        file_dump = open(f"./{phase}_imgs{oneItem}.txt",'w')
        #files = [f for f in files if os.path.isfile(dir+'/'+f)] #Filtering only the files.
        #print(*files, sep="\n")
        
        imgs = {}
        for i in range(len(class_names)):
            imgs[i] = {"images":[],"num":0, 'name': class_names[i], 'box': []}

        for file in files:
            with open(dir+file) as f:
                data = json.load(f)
                items = list(data.keys())
                if not oneItemOnly or (oneItemOnly and len(items)==3): 
                    # 3 because there are always 'source' and 'pair_id' together with items
                    items.remove('source')
                    items.remove('pair_id')
                    for item in items:
                        class_img = data[item]["category_id"]-1
                        file = (file.split(".")[0])+".jpg" # Suffix useful in distribute_classes.py
                        box = data[item]["bounding_box"] 
                        imgs[class_img]["images"].append(file)
                        imgs[class_img]["box"].append(box)  # Useful to crop the images
                        imgs[class_img]["num"] += 1    
                f.close()
        
        #imgs = sorted(imgs, key=lambda x:x.split(',')[1])
        for k in imgs.keys():
            class_stats[k]["num"] += imgs[k]["num"]
        json.dump(imgs,file_dump)
            
        file_dump.close()
        
    file = open(f"./data_stats{oneItem}.txt",'w')
    tot = 0
    for k in class_stats.keys():
        tot += class_stats[k]['num']
    if oneItemOnly:
        s = "IMGS W/1 clothing: {}".format(tot)
    else:
        s = "ITEMS: {}".format(tot)
    file.write(s+"\n")
    print(s)
    print("STATS: {}".format(class_stats))
    json.dump(class_stats,file)
    file.close()

    return

def RemoveDir(dst_path:str):
    """ Remove existing database\n
        Parameters:
            dst_path: root of the dataset"""
    try:
        shutil.rmtree(dst_path)
    except Exception as e:
        print(f'Failed to delete directory: {e}')
        exit()
    return

def ZipDir(src_path:str):
    """ Zip the given directory\n
        Parameters:
            src_path: root of the dataset"""
    try:
        if src_path[-1] == "/":
            src_path = src_path[:-1]
        shutil.make_archive(src_path,'zip',src_path)
    except Exception as e:
        print(f'Failed to compress directory: {e}')
        exit()
    return

def CreateKerasCompatibleDS(num_imgs:list,dst_path:str,crop:bool=False,oneItem:bool=False):
    """Create a dataset format compatible with the Keras 
        tf.keras.utils.image_dataset_from_directory(...) utility\n
        Parameters:
            num_imgs: a list with the number of images for each class. 
            The list can be obtained with the createSizes procedure and has to be 
            ordered according to the class_names array
            dst_path: a string containing the dataset root
            crop: if True only the pixels within the object bounding box are selected
            oneItem: if True create a dataset using only images with 1 clothing\n"""
    
    print("### Starting main dataset creation ###")
    if oneItem:
        oneItem = "_oneItem"
    else:
        oneItem = ""
    dst_path += "/"
    img_id = 1
    for phase in ("validation","train"):
        src_path = "dataset/{}/image/".format(phase)
        try:
            f = "./{}_imgs{}.txt".format(phase,oneItem)
            fp = open(f)
            data = json.load(fp)
        except Exception as e:
            print(e)
            exit()
        finally:
            fp.close()


        # Copy data in keras database directory
        for k in data.keys():
            if(num_imgs[int(k)] == 0): # Class already full
                continue
            class_dst = dst_path+data[k]['name']+"/"
            os.makedirs(class_dst,exist_ok=True)
            for i in range(len(data[k]['images'])):
                src = src_path+data[k]['images'][i]
                new_img = str(img_id).zfill(6)+".jpg"
                dst = class_dst+new_img
                #print(src)
                #print(dst)
                if not crop:
                    shutil.copyfile(src,dst)
                else: # Crop image before copying
                    img = Image.open(src,formats=['JPEG'])
                    #print(data[k]['box'][i])
                    img = img.crop(data[k]['box'][i])
                    img.save(dst,formats=['JPEG'])
                img_id += 1
                num_imgs[int(k)] -= 1
                if(num_imgs[int(k)] == 0): # Class already full
                    print(f"Finished to fill class {k} [{phase} data]")
                    break
    print("IMGS MOVED: {}".format(img_id-1))
    return

def ComputeClassSizes(sizes:list, class_stats:dict):
    """ Compute the size of each class based on the number of images of the dataset.\n
    Parameters:
        sizes: a list with the sizes of the datasets to be created
        class_stats: the second line of the file \"data_stats.txt\" (use the function OneClothingFinder to create it)\n
    Returns:
        mv_imgs: a dictionary with the size as key and a list the class sizes as value\n
    The number of images for each dataset can be slightly higher but never lower than the given size.\n
    Example:
        sizes = [45000] --> sum(mv_imgs[45000]) = 45002"""
    # Order the dict by natural order
    mv_imgs = {}
    tmp = sorted(class_stats.items(), key=lambda x:x[1]['num'])
    class_stats = dict(tmp)
    num_classes = len(class_names)

    for size in sizes: # size is supposed to be <45002
        #print(f"Computing sizes for {size} images dataset")
        mv_imgs[size] = [0]*num_classes
        lower = 1
        num_to_mv = ceil(size/num_classes)
        rm_imgs_to_mv = 0
        rm_classes_to_mv = num_classes
        for k in class_stats.keys():
            class_imgs = class_stats[k]['num']
            class_index = int(k)
            if lower==1 and class_imgs<=num_to_mv:
                mv_imgs[size][class_index] = class_imgs
                rm_imgs_to_mv += class_imgs
                rm_classes_to_mv -= 1
                num_to_mv = ceil((size-rm_imgs_to_mv)/rm_classes_to_mv)
            else:
                mv_imgs[size][class_index] = num_to_mv
                lower=0
    return mv_imgs

def CreateSubsetDS(src_path:str, dst_path_root:str, mv_imgs:dict, compression:str=None):
    """ Create a dataset that are subsets of the main one.\n
    Parameters:
        src_path: the main dataset as string (read notes at the end of the documentation)
        dst_path_root: the datasets names prefix as string
            Example:
                dst_path_root = "ds" will produce datasets with ds_{size} as root directory
        mv_imgs: the output of the ComputeClassSizes function
        compression: 'zip' if you want to create archives\n 
    Outputs:
        Datasets with a subset of elements w.r.t. the main one\n   
    WARNING: be sure you created the main dataset (the one with most images) before proceeding
    since the images are moved from that one.\n
    Example:
        mv_imgs = {5000 : [...]} requires to have an already existing main dataset with more than 5000 images
    """
    src_path += "/"
    if dst_path_root[-1] == "/":
        dst_path_root = dst_path_root[:-1]
    for k in mv_imgs.keys():
        print(f"### Creating dataset with {k} images ###")
        dst_path = "{}_{}/".format(dst_path_root,k)
        for dir in os.listdir(src_path):
            class_index = class_names.index(dir)
            src = src_path+dir+"/"
            dst = dst_path+dir+"/"
            if(mv_imgs[k][class_index]<max(mv_imgs[k])): 
                # Copy the entire dir if the class has less elements than other classes 
                shutil.copytree(src,dst,dirs_exist_ok=True)
            else:
                os.makedirs(dst,exist_ok=True)
                # Copy the necessary images if the class has more images than required 
                for file in os.listdir(src)[:mv_imgs[k][class_index]]:
                    shutil.copyfile(src+file,dst+file)
        if compression == 'zip':
            ZipDir(dst_path)
            RemoveDir(dst_path)
    return


if __name__=="__main__":
    ClothingFinder(False)
    #class_stats = json.loads(open("data_stats_cropped.txt").readlines()[1])
    #num_imgs = ComputeClassSizes([45000],class_stats)
    #RemoveDir("keras_dataset_cropped")
    #CreateKerasCompatibleDS(num_imgs[45000],"keras_dataset_oneItem/",oneItem=True)
    #num_imgs = ComputeClassSizes([5000,10000,20000,30000],class_stats)
    #CreateSubsetDS("keras_dataset","keras_dataset",num_imgs)
    pass
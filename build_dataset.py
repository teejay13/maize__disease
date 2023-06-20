import argparse
import random
import os
import shutil
import numpy as np

from PIL import Image
from tqdm  import tqdm

SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',default='../maizedataset/' , help="Directory with the maize dataset")
#arser.add_argument('--output_dir',default='mzdata', help="Where to write the new data" )
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    
    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
    
    classes_dir = ['HEATHLY','MLN','MSV']
    
    val_ratio = 0.15
    test_ratio =  0.05
    
    for cls  in  classes_dir:
        os.makedirs(args.data_dir +'train/' + cls)
        os.makedirs(args.data_dir +'val/' + cls)
        os.makedirs(args.data_dir +'test/' + cls)
    
        src = args.data_dir + cls # Folder to copy images from

        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        train_FileNames, val_FileNames, test_FileNames =  np.split(np.array(allFileNames),[int(len(allFileNames)* (1 - (val_ratio + test_ratio))),int(len(allFileNames)* (1 - test_ratio))])
    
    
        train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
        val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
        test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
        
        print('Total images: ', len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Validation: ', len(val_FileNames))
        print('Testing: ', len(test_FileNames))

        # Copy-pasting images
        for name in train_FileNames:
            shutil.copy(name, args.data_dir +'train/' + cls)

        for name in val_FileNames:
            shutil.copy(name, args.data_dir +'val/' + cls)

        for name in test_FileNames:
            shutil.copy(name, args.data_dir +'test/' + cls)
        
    
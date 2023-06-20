from util import config
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch


def visualize_batch(batch, classes, dataset_type):
    fig = plt.figure("{} batch".format(dataset_type),figsize=(config.BATCH_SIZE, config.BATCH_SIZE))
    
    for i in range(0, config.BATCH_SIZE):
        ax = plt.subplot(2, 4, i + 1)
        
        image = batch[0][i].cpu().numpy()
        image = image.transpose((1, 2, 0))
        image = (image * 255.0).astype("uint8")
        
        
        idx = batch[1][i]
        label = classes[idx]
        
        plt.imshow(image)
        plt.title(label)
        plt.axis("off")
  
    plt.tight_layout()
    plt.axis("off")
    
    
if __name__ == '__main__':
        
    resize = transforms.Resize(size=(config.INPUT_HEIGHT,
            config.INPUT_WIDTH))
    hFlip = transforms.RandomHorizontalFlip(p=0.25)
    vFlip = transforms.RandomVerticalFlip(p=0.25)
    rotate = transforms.RandomRotation(degrees=15)

    trainTransforms = transforms.Compose([
        resize,hFlip,vFlip,rotate,
        transforms.ToTensor()
    ])

    valTransforms = transforms.Compose([resize, transforms.ToTensor()])
    
    print("[INFO] loading the training and validation dataset...")
    trainDataset = ImageFolder(root=config.MAIZE_DATASET_PATH+"/"+config.TRAIN,
            transform=trainTransforms)
    valDataset = ImageFolder(root=config.MAIZE_DATASET_PATH+"/"+config.VAL, 
            transform=valTransforms)
    print("[INFO] training dataset contains {} samples...".format(
            len(trainDataset)))
    print("[INFO] validation dataset contains {} samples...".format(
            len(valDataset)))
    
    print("[INFO] creating training and validation set dataloaders...")
    trainDataLoader = DataLoader(trainDataset,batch_size=config.BATCH_SIZE, shuffle=True)
    valDataLoader = DataLoader(valDataset, batch_size=config.BATCH_SIZE)
    
    # grab a batch from both training and validation dataloader
    trainBatch = next(iter(trainDataLoader))
    valBatch = next(iter(valDataLoader))
    # visualize the training and validation set batches
    print("[INFO] visualizing training and validation batch...")
    visualize_batch(trainBatch, trainDataset.classes, "train")
    visualize_batch(valBatch, valDataset.classes, "val")
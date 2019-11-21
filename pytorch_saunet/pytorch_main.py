from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#-----------------------------------------------------------

class FormsDataset(Dataset):
    def __init__(self, images, masks, num_classes: int, transforms=None):
        self.images = images
        self.masks = masks
        self.num_classes = num_classes
        self.transforms = transforms
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.astype(np.float32)
        image = np.expand_dims(image, -1)
        image = image / 255
        if self.transforms:
            image = self.transforms(image)
            
        mask = self.masks[idx]
        mask = mask.astype(np.float32)
        mask = mask / 255
        mask[mask > .7] = 1
        mask[mask <= .7] = 0
        if self.transforms:
            mask = self.transforms(mask)
    
        return image, mask
    
    def __len__(self):
        return len(self.images)
train_dataset = FormsDataset(train_images, train_masks, number_of_classes, get_transformations(True))
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f'Train dataset has {len(train_data_loader)} batches of size {batch_size}')



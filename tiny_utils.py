import torch
import torchvision

import numpy as np 
import pandas as pd 

from PIL import Image
import matplotlib.pyplot as plt 
from skimage.morphology import binary_opening, disk, label

def test():
    repr = "MODULE WORKS!!"
    return repr

def make_data(data_loc): 
    data = pd.read_csv(data_loc) 
    data = data.dropna() 
    return data.sample(frac = 0.125, replace=False, random_state=42)  

def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros([shape[0]*shape[1],1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]
    
#create dataset
class ShipDatabaseSegmation(torch.utils.data.Dataset):
    def __init__(self,in_df,root_path,transforms=None):
        imagesIds = in_df['ImageId'].tolist()
        self.image_ids =  list(set(imagesIds))
        self.in_df = in_df
        self.root_path = root_path
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self,idx):
        ImageId = self.image_ids[idx]
        img = Image.open(self.root_path + "/"+ ImageId)
        img_masks = self.in_df.loc[self.in_df['ImageId'] == ImageId, 'EncodedPixels'].tolist()
        all_masks = np.zeros((768, 768))
        for mask in img_masks:
            all_masks += rle_decode(mask)
        
        #all_masks = np.expand_dims(all_masks,axis=0)
        if self.transforms is not None:
            img = self.transforms(img)
            all_masks = self.transforms(all_masks)

        return img,all_masks
    
def validation_pred(root_path,image_names,model,device):
    out_pred_rows = []
    for img_name in image_names:
        c_img = Image.open(root_path+"/"+img_name)
        covnertTensor = torchvision.transforms.transforms.ToTensor()
        c_img = covnertTensor(c_img)
        c_img = c_img.unsqueeze(0)
        if device == 'cuda':
            c_img = c_img.cuda()
        cur_seg = model(c_img)
        cur_seg = cur_seg.squeeze(0).squeeze(0).detach().cpu().numpy()
        cur_seg[cur_seg < 0.5] = 0 
        cur_seg[cur_seg >= 0.5] = 1
        cur_rles = multi_rle_encode(cur_seg,max_mean_threshold=1.0)
        if len(cur_rles)>0:
            for c_rle in cur_rles:
                out_pred_rows += [{'ImageId': img_name, 'EncodedPixels': c_rle}]
        else:
            out_pred_rows += [{'ImageId': img_name, 'EncodedPixels': None}]

    return out_pred_rows
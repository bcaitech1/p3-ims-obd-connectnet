import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os 
import numpy as np
import cv2
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from preprocess  import get_category
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F

__all__=["CustomDataLoader"]


size = 512

#https://www.kaggle.com/haqishen/gridmask
class GridMask(DualTransform):
    """GridMask aug#mentation for image classification and object detection.
    
    Author: Qishen Ha
    Email: haqishen@gmail.com
    2020/01/29

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value
                
                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')
        

def data_norm():
  return A.Compose([
      A.Resize(size, size),  
      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),                       
      ToTensorV2()
      ])

    
def data_basic():
  return A.Compose([
      A.Resize(size, size),                       
      ToTensorV2()
      ])

def data_hflip():
  return A.Compose([
    A.Resize(size, size), 
    A.HorizontalFlip(),
    ToTensorV2(transpose_mask=True)
    ])

def data_rotate():
  return A.Compose([
    A.Resize(size, size),                            
    A.Rotate(limit=65),
    ToTensorV2(transpose_mask=True)
    ])

def data_bright():
  return A.Compose([
    A.Resize(size, size),                            
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
    ToTensorV2(transpose_mask=True)
    ])


def data_gridmask():
  return A.Compose([
    A.Resize(size, size),                           
    GridMask(num_grid=(5), p=0.5, rotate=25),
    ToTensorV2(transpose_mask=True)
    ])

def data_shift():
  return A.Compose([
    A.Resize(size, size),                           
    A.ShiftScaleRotate(),
    ToTensorV2(transpose_mask=True)
    ])

def data_snow():
  return A.Compose([
    A.Resize(size, size),                           
    A.RandomSnow(),
    ToTensorV2(transpose_mask=True)
    ])


def data_test():
  return A.Compose([
    A.Resize(size, size),  
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
    A.RandomResizedCrop(512,512,scale = (0.6,0.8)),
    A.VerticalFlip(p=0.5),
    A.OneOf([
            A.OpticalDistortion(p=0.45),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.45)
        ], p=0.5),
    ToTensorV2()
  ])

def data_total():
  return A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
    A.RandomResizedCrop(512,512,scale = (0.5,0.8)),
    A.VerticalFlip(p=0.5),
    A.OneOf([
            A.OpticalDistortion(p=0.45),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.45)
        ], p=0.5),
    ToTensorV2()
  ])

def data_total_dist():
  return A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
    A.RandomResizedCrop(512,512,scale = (0.5,0.8)),
    A.OpticalDistortion(),
    A.VerticalFlip(p=0.5),
    ToTensorV2()
  ])


def data_mask_crop1():
  return A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
    A.CropNonEmptyMaskIfExists(height=200, width=200),
    A.Resize(256, 256),
    ToTensorV2()
  ])

def data_mask_crop11():
  return A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
    A.OneOf([
      A.RandomResizedCrop(512,512,scale = (0.5,0.8),p=0.8),
      A.CropNonEmptyMaskIfExists(height=300, width=300, p=0.2),
    ], p=0.5),
    A.Resize(512, 512),
    A.OpticalDistortion(p=0.2),
    A.VerticalFlip(p=0.5),
    ToTensorV2()
  ])

def data_mask_crop4():
  return A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
    A.OneOf([
      A.RandomResizedCrop(512,512,scale = (0.5,0.8),p=0.8),
      A.CropNonEmptyMaskIfExists(height=300, width=300, p=0.2),
    ], p=0.5),
    A.Resize(512, 512),
    A.VerticalFlip(p=0.5),
    ToTensorV2()
  ])

def data_clahe():
  return A.Compose([
    A.CLAHE(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
    A.OneOf([
      A.RandomResizedCrop(512,512,scale = (0.5,0.8),p=0.8),
      A.CropNonEmptyMaskIfExists(height=300, width=300, p=0.2),
    ], p=0.5),
    A.Resize(512, 512),
    A.VerticalFlip(p=0.5),
    ToTensorV2()
    ])

def data_mask_crop3():
  return A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
    A.CropNonEmptyMaskIfExists(height=256, width=256),
    A.Resize(512, 512),
    A.VerticalFlip(p=0.5),
    ToTensorV2()
  ])


def data_sample():
  return A.Compose([
    A.SmallestMaxSize(max_size=200),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.RandomCrop(height=size, width=size),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
  ])




class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir,  mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.category_names = get_category()
    

    def get_classname(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"


        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        dataset_path = '/opt/ml/input/data'
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        #images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = self.get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            
            return images, image_infos
    
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())

if __name__ =="__main__":
    ## Dataset Test code
    dataset_path = '/opt/ml/input/data'
    train_path = dataset_path + '/train.json'
    test_path = dataset_path + '/test.json'
    val_path = dataset_path + '/val.json'
   
    category_names = get_category()
    train_transform = globals()['data_mask_crop1']()
    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)
    print(len(train_dataset))
    img, masks, image_infos = train_dataset[1]
    print(image_infos)


    test_transform = globals()['data_total']()
    test_dataset = CustomDataLoader(data_dir=test_path,   mode='test', transform=test_transform)
    print(len(test_dataset))
    img, image_infos = test_dataset[1]
    print(image_infos)

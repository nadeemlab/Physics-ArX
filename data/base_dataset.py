"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from abc import ABC, abstractmethod
import torch

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
        
def transform_cbct_ct_pair(ct_arr, cbct_arr, eso_arr, gtv_arr, dose_arr, apply_transform=False):
  """Apply several (same) transforms to ct/cbctlabel pair which are numpy arrays.
      Use nearest neighbor interpolation for Label
  """
  # Convert to PIL image
  ct_pil = TF.to_pil_image(ct_arr, mode='F')
  cbct_pil = TF.to_pil_image(cbct_arr, mode='F')
  dose_pil = TF.to_pil_image(dose_arr, mode='F')
  gtv_pil = TF.to_pil_image(gtv_arr*255)
  eso_pil = TF.to_pil_image(eso_arr*255)
  #print("T start: ", label_arr.max())
  #print("T start: ", label_pil.getextrema())
  
  if apply_transform is True:
    
    if random.random() > 0.5:
      # Horizontal flip
      if random.random() > 0.5:
        ct_pil = TF.hflip(ct_pil)
        cbct_pil = TF.hflip(cbct_pil)
        dose_pil = TF.hflip(dose_pil)
        gtv_pil = TF.hflip(gtv_pil)
        eso_pil = TF.hflip(eso_pil)
      
      # Vertical flip
      if random.random() > 0.5:
        ct_pil = TF.vflip(ct_pil)
        cbct_pil = TF.vflip(cbct_pil)
        dose_pil = TF.vflip(dose_pil)
        gtv_pil = TF.vflip(gtv_pil)
        eso_pil = TF.vflip(eso_pil)
      
      # Rotation +/- 35 degrees
      if random.random() > 0.5:
        angle = random.uniform(-35.0, 35.0)
        ct_pil   = TF.affine(ct_pil, angle=angle, translate=(0,0), scale=1, shear=(0,0), resample=Image.BILINEAR)
        cbct_pil = TF.affine(cbct_pil, angle=angle, translate=(0,0), scale=1, shear=(0,0), resample=Image.BILINEAR)
        dose_pil = TF.affine(dose_pil, angle=angle, translate=(0,0), scale=1, shear=(0,0), resample=Image.BILINEAR)
        gtv_pil = TF.affine(gtv_pil, angle=angle, translate=(0,0), scale=1, shear=(0,0), resample=Image.NEAREST)
        eso_pil = TF.affine(eso_pil, angle=angle, translate=(0,0), scale=1, shear=(0,0), resample=Image.NEAREST)
        
      
      # Scale 0.8 -- 1.2
      if random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        ct_pil   = TF.affine(ct_pil, angle=0, translate=(0,0), scale=scale, shear=(0,0), resample=Image.BILINEAR)
        cbct_pil = TF.affine(cbct_pil, angle=0, translate=(0,0), scale=scale, shear=(0,0), resample=Image.BILINEAR)
        dose_pil = TF.affine(dose_pil, angle=0, translate=(0,0), scale=scale, shear=(0,0), resample=Image.BILINEAR)
        gtv_pil = TF.affine(gtv_pil, angle=0, translate=(0,0), scale=scale, shear=(0,0), resample=Image.NEAREST)
        eso_pil = TF.affine(eso_pil, angle=0, translate=(0,0), scale=scale, shear=(0,0), resample=Image.NEAREST)
      
      # translate -10 to 10 pixels
      if random.random() > 0.5:
        h = random.uniform(-10, 10)
        w = random.uniform(-10, 10)
        ct_pil   = TF.affine(ct_pil, angle=0, translate=(w,h), scale=1, shear=(0,0), resample=Image.BILINEAR)
        cbct_pil = TF.affine(cbct_pil, angle=0, translate=(w,h), scale=1, shear=(0,0), resample=Image.BILINEAR)
        dose_pil = TF.affine(dose_pil, angle=0, translate=(w,h), scale=1, shear=(0,0), resample=Image.BILINEAR)
        gtv_pil = TF.affine(gtv_pil, angle=0, translate=(w,h), scale=1, shear=(0,0), resample=Image.NEAREST)
        eso_pil = TF.affine(eso_pil, angle=0, translate=(w,h), scale=1, shear=(0,0), resample=Image.NEAREST)
      
      # Shear +/- 8 degrees
      if random.random() > 0.5:
        x = random.uniform(-8, 8)
        y = random.uniform(-8, 8)
        ct_pil   = TF.affine(ct_pil, angle=0, translate=(0,0), scale=1, shear=(x,y), resample=Image.BILINEAR)
        cbct_pil = TF.affine(cbct_pil, angle=0, translate=(0,0), scale=1, shear=(x,y), resample=Image.BILINEAR)
        dose_pil = TF.affine(dose_pil, angle=0, translate=(0,0), scale=1, shear=(x,y), resample=Image.BILINEAR)
        gtv_pil = TF.affine(gtv_pil, angle=0, translate=(0,0), scale=1, shear=(x,y), resample=Image.NEAREST)
        eso_pil = TF.affine(eso_pil, angle=0, translate=(0,0), scale=1, shear=(x,y), resample=Image.NEAREST)
    
  ct_arr   = TF.to_tensor(ct_pil)
  cbct_arr = TF.to_tensor(cbct_pil)
  dose_arr = TF.to_tensor(dose_pil)
  gtv_arr = TF.to_tensor(gtv_pil)
  eso_arr = TF.to_tensor(eso_pil)
  
  # Rescale [0 1] values to [-1 1] range        m - 0
  #                                    m --->  ------- * (1 - (-1)) + (-1)
  #                                             1 - 0
  ct_arr = ct_arr * 2 - 1
  cbct_arr = cbct_arr * 2 - 1
  dose_arr = dose_arr * 2 - 1
  
  #print("T end: ", label_arr.max())
  return ct_arr, cbct_arr, eso_arr, gtv_arr, dose_arr

def transform_cbct_ct_pair3D(ct_arr, cbct_arr, rt_arr, apply_transform=False):
  """Apply several (same) transforms to ct/cbct label pair which are numpy arrays.
      Use nearest neighbor interpolation for Label
  """
  
  ct_arr = torch.from_numpy(ct_arr)
  cbct_arr = torch.from_numpy(cbct_arr)
  rt_arr = torch.from_numpy((rt_arr).astype(np.float32))
  
  if apply_transform is True:
      if random.random() < 0.5:  # 50 % chance of applying random transformation

        scale, angle, h, w, x, y = 1, 0, 0, 0, 0, 0

        # Rotation +/- 25 degrees
        if random.random() < 0.5:
          angle = random.uniform(-25.0, 25.0)
        # Scale 0.8/1.2
        if random.random() < 0.5:
          scale = random.uniform(0.8, 1.2)
        # translate -10 to 10 pixels
        if random.random() < 0.5:
          h = random.uniform(-10, 10)
          w = random.uniform(-10, 10)
        # Shear +/- 8 degrees
        if random.random() < 0.5:
          x = random.uniform(-8, 8)
          y = random.uniform(-8, 8)

        depth = ct_arr.shape[0]
        
        for i in range(depth):
            
            ct_pil = TF.to_pil_image(ct_arr[i], mode='F')
            cbct_pil = TF.to_pil_image(cbct_arr[i], mode='F')
            rt_pil = TF.to_pil_image(rt_arr[i], mode='F')
            
            ct_pil   = TF.affine(ct_pil, angle=angle, translate=(w,h), scale=scale, shear=(x,y), resample=Image.BILINEAR)
            cbct_pil = TF.affine(cbct_pil, angle=angle, translate=(w,h), scale=scale, shear=(x,y), resample=Image.BILINEAR)
            rt_pil = TF.affine(rt_pil, angle=angle, translate=(w,h), scale=scale, shear=(x,y), resample=Image.NEAREST)
            
            ct_arr[i] = TF.to_tensor(ct_pil)
            cbct_arr[i] = TF.to_tensor(cbct_pil)
            rt_arr[i] = TF.to_tensor(rt_pil)
  
  # Rescale [0 1] values to [-1 1] range        m - 0
  #                                    m --->  ------- * (1 - (-1)) + (-1)
  #                                             1 - 0
  ct_arr = ct_arr * 2. - 1.
  cbct_arr = cbct_arr * 2. - 1.
  rt_arr = (rt_arr/4.0) * 2 - 1 # Rtstruct are in [0 4] range (not using GTV)
  
  return ct_arr, cbct_arr, rt_arr

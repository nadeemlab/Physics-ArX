import numpy as np
import os
import re
import nrrd
from collections import OrderedDict
import fnmatch
from skimage.transform import resize
import imgaug as ia
import imgaug.augmenters as iaa

def augmenter_test(Images, Masks, image_names, mask_names, out_folder):
    seed = np.random.randint(0, 2 ** 16)
    print(seed)
    ia.seed(seed)
    seq1 = iaa.Sequential([iaa.SigmoidContrast(gain=(10), cutoff=(0.6)), ]).to_deterministic()
    seq2 = iaa.Sequential([iaa.Sharpen(alpha=(1.0), lightness=(4.0)), ]).to_deterministic()
    seq3 = iaa.Sequential([iaa.Affine(scale=(1.0, 1.3), rotate=(-10, 0))]).to_deterministic()
    seq4 = iaa.Sequential([iaa.Affine(scale=(1.0, 1.3), rotate=(0, 10), )]).to_deterministic()
    seq5 = iaa.Sequential([iaa.Affine(scale=(0.8, 1.0), rotate=(-10, 0), )]).to_deterministic()
    seq6 = iaa.Sequential([iaa.Affine(scale=(0.8, 1.0), rotate=(0, 10), )]).to_deterministic()
    seq7 = iaa.Sequential([iaa.Affine(shear=(-20, 0))]).to_deterministic()
    seq8 = iaa.Sequential([iaa.Affine(shear=(0, 20))]).to_deterministic()

    t1_info = OrderedDict()
    t1_info['space'] = 'left-posterior-superior'
    t1_info['space directions'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    t1_info['space origin'] = [0, 0, 0]
    paths = []
    for i in range(1, 9):
        pth = out_folder + "_{}".format(i)  # for psCBCT
        os.makedirs(pth, exist_ok=True)
        paths.append(pth)

    for img, name in zip(Images, image_names):

        img_aug = seq1(image=img)
        nrrd.write(paths[0] + "/" + name, img_aug, t1_info)

        img_aug = seq2(image=img)
        nrrd.write(paths[1] + "/" + name, img_aug, t1_info)

        img_aug = seq3(image=img)
        nrrd.write(paths[2] + "/" + name, img_aug, t1_info)

        img_aug = seq4(image=img)
        nrrd.write(paths[3] + "/" + name, img_aug, t1_info)

        img_aug = seq5(image=img)
        nrrd.write(paths[4] + "/" + name, img_aug, t1_info)

        img_aug = seq6(image=img)
        nrrd.write(paths[5] + "/" + name, img_aug, t1_info)

        img_aug = seq7(image=img)
        nrrd.write(paths[6] + "/" + name, img_aug, t1_info)

        img_aug = seq8(image=img)
        nrrd.write(paths[7] + "/" + name, img_aug, t1_info)

    for mask, name in zip(Masks, mask_names):
        nrrd.write(paths[0] + "/" + name, mask, t1_info)  # seq1(SigmoidContrast) and seq2(Sharpen) are not applied to masks
        nrrd.write(paths[1] + "/" + name, mask, t1_info)

        img_aug, mask_aug = seq3(images=[Images[0]], segmentation_maps=[mask])
        nrrd.write(paths[2] + "/" + name, mask_aug[0], t1_info)

        img_aug, mask_aug = seq4(images=[Images[0]], segmentation_maps=[mask])
        nrrd.write(paths[3] + "/" + name, mask_aug[0], t1_info)

        img_aug, mask_aug = seq5(images=[Images[0]], segmentation_maps=[mask])
        nrrd.write(paths[4] + "/" + name, mask_aug[0], t1_info)

        img_aug, mask_aug = seq6(images=[Images[0]], segmentation_maps=[mask])
        nrrd.write(paths[5] + "/" + name, mask_aug[0], t1_info)

        img_aug, mask_aug = seq7(images=[Images[0]], segmentation_maps=[mask])
        nrrd.write(paths[6] + "/" + name, mask_aug[0], t1_info)

        img_aug, mask_aug = seq8(images=[Images[0]], segmentation_maps=[mask])
        nrrd.write(paths[7] + "/" + name, mask_aug[0], t1_info)


def load_for_aug():

    path = "./train_psCBCT_AAPM/" #_1 &_5
    for i in os.listdir(path):  # folder

        if '_1' in i or '_5' in i:
            filenames = os.listdir(path + "/" + i + "/")

            img_names = []  # CT and CBCT (or psCBCT)
            mask_names = []  # Spinal Cord, Esophasgus, Heart, Lungs; binary masks
            images = []  # CT and CBCT (or psCBCT)
            masks = []  # Spinal Cord, Esophasgus, Heart, Lungs; binary masks

            masks_suffixes = ['_Eso', 'Cord', 'Heart', 'Lungs']
            for name in filenames:
                is_mask = False
                for suffix in masks_suffixes:
                    if suffix in name:
                        is_mask = True
                img, img_info = nrrd.read(path + "/" + i + "/" + name)

                if is_mask is True:
                    img = resize(img, (128, 128, 128), order=0, preserve_range=True, anti_aliasing=False)  # Nearest Neighbor interpolation
                    img = img.astype(np.int32)
                    #img = np.reshape(img, [128, 128, 128, 1])
                else:
                    img = resize(img, (128, 128, 128), order=1, preserve_range=True, anti_aliasing=False)  # Linear interpolation
                #img = np.reshape(img, [1, 128, 128, 128, 1])

                if is_mask is True:
                    mask_names.append(name)
                    masks.append(img)
                else:
                    img_names.append(name)
                    images.append(img)
                    if '_pCT_OSSART_' in name:
                        out_folder_name = name.split('.nrrd')[0]
            out_folder_name = './augmented/' + out_folder_name
            
            augmenter_test(images, masks, img_names, mask_names, out_folder_name)

if __name__ == "__main__":
    load_for_aug()
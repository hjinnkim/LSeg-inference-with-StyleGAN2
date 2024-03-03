import os
import dnnlib
import numpy as np
from tqdm import tqdm
import click
import torch

import clip_text

from modules.lseg_module import LSegModule

from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms as transforms

import time


def get_dataset(path: str):
    training_set_kwargs={
        "class_name": "training.dataset.ImageFolderDataset",
        "path": path,
        "use_labels": False,
        "max_size": 10000,
        "xflip": False,
        "resolution": 256
    }
    dataset = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    return dataset

@click.command()
@click.option('--dataset', help='Path to the text file', required=True)
@click.option('--dataset_path', help='Word to extract related sentences', required=True)
@click.option('--save_path', help='Word to extract related sentences', required=True)
@click.option('--flip', default=False, is_flag=True)
@click.option('--mask', default=False, is_flag=True)
@click.option('--no-seg', default=False, is_flag=True)
@click.option('--fp16', default=False, is_flag=True)

def main(dataset, dataset_path, save_path, flip, mask, no_seg, fp16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = dataset
    dataset_path = dataset_path
    save_path = save_path
    flip = flip
    mask = mask
    no_seg = no_seg
    fp16 = fp16
    assert not no_seg or mask
        
    if flip:
        save_path+='-flip'
    if mask:
        save_path+='-mask'
    if no_seg:
        save_path+='-no_seg'
    if fp16:
        save_path+='-fp16'
    
    if 'ffhq' in dataset:
        classes = clip_text.FFHQ_CLASSES
        background_classes = clip_text.FFHQ_BACKGROUND_CLASSES
        print('class: FFHQ_CLASSES')
        print()
        print(classes)
        print()
    elif 'lsun_bedroom' in dataset:
        classes = clip_text.LSUNBEDROOM_CLASSES
        if 'lsun_bedroom2' in dataset:
            classes = clip_text.LSUNBEDROOM_BASE_CLASSES2
            print('class: LSUNBEDROOM_CLASSES2')
            print()
        else:
            print('class: LSUNBEDROOM_CLASSES')
            print()
        background_classes = clip_text.LSUNBEDROOM_BACKGROUND_CLASSES
        print(classes)  
        print() 
    elif 'lsun_church' in dataset:
        classes = clip_text.LSUNCHURCH_CLASSES
        background_classes = clip_text.LSUNCHURCH_BACKGROUND_CLASSES
        print('class: LSUNCHURCH_CLASSES')
        print()
        print(classes)   
        print()
    # elif 'lsun_cat' in dataset:
    #     classes = clip_text.LSUNCAT_CLASSES
    #     print('class: LSUNCAT_CLASSES')
        # print(classes)   
    # elif 'afhq_dog' in dataset:
    #     classes = clip_text.AFHQDOG_CLASSES
    #     print('class: AFHQDOG_CLASSES')
        # print(classes)   
    elif 'afhq_wild' in dataset:
        classes = clip_text.AFHQWILD_CLASSES
        background_classes = clip_text.AFHQWILD_BACKGROUND_CLASSES
        print('class: AFHQWILD_CLASSES')
        print()
        print(classes)   
        print()
    elif 'afhq_cat' in dataset:
        classes = clip_text.AFHQCAT_CLASSES
        background_classes = clip_text.AFHQCAT_BACKGROUND_CLASSES
        print('class: AFHQCAT_CLASSES')
        print()
        print(classes)   
        print()
    
    print('Num of classes: ', len(classes))
    print('Num of background: ', len(background_classes))
    print()
    
    print("Options   ::   mask: ", mask, "   ||   segmentation: ", not no_seg, "   ||   flip: ", flip, "   ||   fp16:", fp16)
    print()
    
    dataset = get_dataset(dataset_path)
    
    def get_new_pallete(num_cls):
        n = num_cls
        pallete = [0]*(n*3)
        for j in range(0,n):
                lab = j
                pallete[j*3+0] = 0
                pallete[j*3+1] = 0
                pallete[j*3+2] = 0
                i = 0
                while (lab > 0):
                        pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                        pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                        pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                        i = i + 1
                        lab >>= 3
        return pallete

    def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None):
        """Get image color pallete for visualizing masks"""
        # put colormap
        out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
        out_img.putpalette(new_palette)

        if out_label_flag:
            assert labels is not None
            u_index = np.unique(npimg)
            patches = []
            for i, index in enumerate(u_index):
                label = labels[index]
                cur_color = [new_palette[index * 3] / 255.0, new_palette[index * 3 + 1] / 255.0, new_palette[index * 3 + 2] / 255.0]
                red_patch = mpatches.Patch(color=cur_color, label=label)
                patches.append(red_patch)
        return out_img, patches

    torch.manual_seed(1)
    
    module = LSegModule(
        backbone='clip_vitl16_384',
        block_depth=0,
        activation='lrelu',
        resolution=256
    ).cuda().eval()
    
    
    # weights = 'checkpoints/demo_e200.ckpt'
    # pretrained_weights = torch.load(weights, map_location='cpu')
    # module.load_state_dict(pretrained_weights['state_dict'], strict=False)
    # torch.save(pretrained_weights['state_dict'], 'lseg.ckpt')
    # exit()
    
    weights = 'lseg.ckpt'
    pretrained_weights = torch.load(weights, map_location='cpu')
    module.load_state_dict(pretrained_weights, strict=False)
    
    num_classes = len(classes)
    labels = classes+background_classes
        
    transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
    os.makedirs(save_path, exist_ok=True)
    
    module.set_imshape()
    with torch.no_grad():
        text_features = module.forward_text_only(labels, device)
    
    if fp16:
        module.half()
        text_features = text_features.half()
    
    gap = np.zeros((256, 2, 3), dtype=np.uint8)
    
    
    t1 = time.time()
    for idx, (img, label) in enumerate(tqdm(dataset)):
        img_t = transform(img.transpose(1, 2, 0)).unsqueeze(0).cuda()
        
        if fp16:
            img_t = img_t.half()
        
        with torch.no_grad():
            # No Flip
            # img_features = module.forward_image_only(img_t)
            # outputs = module.forward_logits(img_features, text_features)
            
            # No Flip
            outputs = module(img_t, text_features, flip)
            # FLIP
            # outputs = module(img_t, text_features, flip=True)
            
            predict = torch.max(outputs, 1, keepdim=True)[1].cpu().numpy()
        
        if mask:
            _mask = predict<num_classes
         
        # with torch.no_grad():
        #     outputs = module.evaluate_random(img_t, labels) 
        #     predict = torch.max(outputs, 1)[1].cpu().numpy()
    

        if no_seg:
            img = img.transpose(1, 2, 0)
            _mask = np.tile(_mask[0].transpose(1, 2, 0), (1, 1, 3))
            imgs = [img, gap, _mask*255, gap, img*_mask]     
            concat_image = np.concatenate(imgs, axis=1).astype(np.uint8)
            concat_image = cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{save_path}/{idx:06}.jpg', concat_image)
        else:
            new_palette = get_new_pallete(len(labels))
            mask, patches = get_new_mask_pallete(predict, new_palette, out_label_flag=True, labels=labels)
            # seg = mask.convert("RGBA")
            seg = np.array(mask.convert("RGB"))
            
            img = img.transpose(1, 2, 0)
            imgs = [img, gap, seg]
            
            if mask:
                _mask = np.tile(_mask[0].transpose(1, 2, 0), (1, 1, 3))
                imgs+=[gap, _mask*255, gap, img*_mask]
            
            concat_image = np.concatenate(imgs, axis=1)
            
            plt.figure()
            plt.imshow(concat_image)
            # plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.5, 1), prop={'size': 20})
            plt.legend(handles=patches, loc='lower left', bbox_to_anchor=(0, 0), prop={'size': 5})
            plt.axis('off')
            # plt.savefig(f'{save_path}/{idx:06}.jpg', concat_image)
            plt.savefig(f'{save_path}/{idx:06}.jpg', bbox_inches='tight')
            plt.cla()   # clear the current axes
            plt.clf()   # clear the current figure
            plt.close() # closes the current figure

    t2 = time.time()
    spent_time = t2-t1
    print('='*100)
    print(spent_time, 'seconds   ||   ', spent_time/60, 'minutes   ||   ', spent_time/60/60, 'hours')
    print('='*100)
    print()
    print()   
if __name__=='__main__':
    main()
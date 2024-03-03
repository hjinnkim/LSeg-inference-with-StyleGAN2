#!/bin/sh

python generate_cams_dataset.py --dataset ffhq --dataset_path /media/khj/T7/datasets/converted/ffhq256x256.zip --save_path segmentation/ffhq --mask
python generate_cams_dataset.py --dataset ffhq --dataset_path /media/khj/T7/datasets/converted/ffhq256x256.zip --save_path segmentation/ffhq --mask --no-seg
python generate_cams_dataset.py --dataset ffhq --dataset_path /media/khj/T7/datasets/converted/ffhq256x256.zip --save_path segmentation/ffhq --mask --no-seg --flip
python generate_cams_dataset.py --dataset ffhq --dataset_path /media/khj/T7/datasets/converted/ffhq256x256.zip --save_path segmentation/ffhq --mask --fp16
python generate_cams_dataset.py --dataset ffhq --dataset_path /media/khj/T7/datasets/converted/ffhq256x256.zip --save_path segmentation/ffhq --mask --no-seg --fp16
python generate_cams_dataset.py --dataset ffhq --dataset_path /media/khj/T7/datasets/converted/ffhq256x256.zip --save_path segmentation/ffhq --mask --no-seg --flip --fp16

python generate_cams_dataset.py --dataset afhq_cat --dataset_path /media/khj/T7/datasets/converted/afhq-cat5k256x256.zip --save_path segmentation/afhq_cat --mask
python generate_cams_dataset.py --dataset afhq_cat --dataset_path /media/khj/T7/datasets/converted/afhq-cat5k256x256.zip --save_path segmentation/afhq_cat --mask --no-seg
python generate_cams_dataset.py --dataset afhq_cat --dataset_path /media/khj/T7/datasets/converted/afhq-cat5k256x256.zip --save_path segmentation/afhq_cat --mask --no-seg --flip
python generate_cams_dataset.py --dataset afhq_cat --dataset_path /media/khj/T7/datasets/converted/afhq-cat5k256x256.zip --save_path segmentation/afhq_cat --mask --fp16
python generate_cams_dataset.py --dataset afhq_cat --dataset_path /media/khj/T7/datasets/converted/afhq-cat5k256x256.zip --save_path segmentation/afhq_cat --mask --no-seg --fp16
python generate_cams_dataset.py --dataset afhq_cat --dataset_path /media/khj/T7/datasets/converted/afhq-cat5k256x256.zip --save_path segmentation/afhq_cat --mask --no-seg --flip --fp16


python generate_cams_dataset.py --dataset afhq_wild --dataset_path /media/khj/T7/datasets/converted/afhq-wild5k256x256.zip --save_path segmentation/afhq_wild --mask
python generate_cams_dataset.py --dataset afhq_wild --dataset_path /media/khj/T7/datasets/converted/afhq-wild5k256x256.zip --save_path segmentation/afhq_wild --mask --no-seg
python generate_cams_dataset.py --dataset afhq_wild --dataset_path /media/khj/T7/datasets/converted/afhq-wild5k256x256.zip --save_path segmentation/afhq_wild --mask --no-seg --flip
python generate_cams_dataset.py --dataset afhq_wild --dataset_path /media/khj/T7/datasets/converted/afhq-wild5k256x256.zip --save_path segmentation/afhq_wild --mask --fp16
python generate_cams_dataset.py --dataset afhq_wild --dataset_path /media/khj/T7/datasets/converted/afhq-wild5k256x256.zip --save_path segmentation/afhq_wild --mask --no-seg --fp16
python generate_cams_dataset.py --dataset afhq_wild --dataset_path /media/khj/T7/datasets/converted/afhq-wild5k256x256.zip --save_path segmentation/afhq_wild --mask --no-seg --flip --fp16


python generate_cams_dataset.py --dataset lsun_church --dataset_path /media/khj/T7/datasets/converted/lsunchurch256x256centercrop126k.zip --save_path segmentation/lsun_church --mask
python generate_cams_dataset.py --dataset lsun_church --dataset_path /media/khj/T7/datasets/converted/lsunchurch256x256centercrop126k.zip --save_path segmentation/lsun_church --mask --no-seg
python generate_cams_dataset.py --dataset lsun_church --dataset_path /media/khj/T7/datasets/converted/lsunchurch256x256centercrop126k.zip --save_path segmentation/lsun_church --mask --no-seg --flip
python generate_cams_dataset.py --dataset lsun_church --dataset_path /media/khj/T7/datasets/converted/lsunchurch256x256centercrop126k.zip --save_path segmentation/lsun_church --mask --fp16
python generate_cams_dataset.py --dataset lsun_church --dataset_path /media/khj/T7/datasets/converted/lsunchurch256x256centercrop126k.zip --save_path segmentation/lsun_church --mask --no-seg --fp16
python generate_cams_dataset.py --dataset lsun_church --dataset_path /media/khj/T7/datasets/converted/lsunchurch256x256centercrop126k.zip --save_path segmentation/lsun_church --mask --no-seg --flip --fp16


python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom --mask
python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom --mask --no-seg
python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom --mask --no-seg --flip
python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom --mask --fp16
python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom --mask --no-seg --fp16
python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom --mask --no-seg --flip --fp16
# 
# python generate_cams_dataset.py --dataset lsun_bedroom2 --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom2 --mask
# python generate_cams_dataset.py --dataset lsun_bedroom2 --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom2 --mask --no-seg
# python generate_cams_dataset.py --dataset lsun_bedroom2 --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom2 --mask --no-seg --flip
# python generate_cams_dataset.py --dataset lsun_bedroom2 --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom2 --mask --fp16
# python generate_cams_dataset.py --dataset lsun_bedroom2 --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom2 --mask --no-seg --fp16
# python generate_cams_dataset.py --dataset lsun_bedroom2 --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom2 --mask --no-seg --flip --fp16


# 
# python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom2 --text 'furnigure, other'
# 

# python generate_cams_dataset.py --dataset afhq_dog --dataset_path /media/khj/portable-hdd1/dataset/converted/afhq-dog5k256x256.zip  --save_path afhq_dog

# python generate_cams_dataset.py --dataset lsun_cat --dataset_path /media/khj/portable-hdd1/dataset/converted/lsuncat200k256x256.zip  --save_path lsun_cat
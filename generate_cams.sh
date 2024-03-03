#!/bin/sh

python generate_cams_dataset.py --dataset ffhq --dataset_path /path/to/ffhq_dataset --save_path segmentation/ffhq --mask
python generate_cams_dataset.py --dataset ffhq --dataset_path /path/to/ffhq_dataset --save_path segmentation/ffhq --mask --no-seg
python generate_cams_dataset.py --dataset ffhq --dataset_path /path/to/ffhq_dataset --save_path segmentation/ffhq --mask --no-seg --flip
python generate_cams_dataset.py --dataset ffhq --dataset_path /path/to/ffhq_dataset --save_path segmentation/ffhq --mask --fp16
python generate_cams_dataset.py --dataset ffhq --dataset_path /path/to/ffhq_dataset --save_path segmentation/ffhq --mask --no-seg --fp16
python generate_cams_dataset.py --dataset ffhq --dataset_path /path/to/ffhq_dataset --save_path segmentation/ffhq --mask --no-seg --flip --fp16


python generate_cams_dataset.py --dataset afhq_cat --dataset_path /path/to/afhq_cat_dataset --save_path segmentation/afhq_cat --mask
python generate_cams_dataset.py --dataset afhq_cat --dataset_path /path/to/afhq_cat_dataset --save_path segmentation/afhq_cat --mask --no-seg
python generate_cams_dataset.py --dataset afhq_cat --dataset_path /path/to/afhq_cat_dataset --save_path segmentation/afhq_cat --mask --no-seg --flip
python generate_cams_dataset.py --dataset afhq_cat --dataset_path /path/to/afhq_cat_dataset --save_path segmentation/afhq_cat --mask --fp16
python generate_cams_dataset.py --dataset afhq_cat --dataset_path /path/to/afhq_cat_dataset --save_path segmentation/afhq_cat --mask --no-seg --fp16
python generate_cams_dataset.py --dataset afhq_cat --dataset_path /path/to/afhq_cat_dataset --save_path segmentation/afhq_cat --mask --no-seg --flip --fp16


python generate_cams_dataset.py --dataset afhq_wild --dataset_path /path/to/afhq_wild_dataset --save_path segmentation/afhq_wild --mask
python generate_cams_dataset.py --dataset afhq_wild --dataset_path /path/to/afhq_wild_dataset --save_path segmentation/afhq_wild --mask --no-seg
python generate_cams_dataset.py --dataset afhq_wild --dataset_path /path/to/afhq_wild_dataset --save_path segmentation/afhq_wild --mask --no-seg --flip
python generate_cams_dataset.py --dataset afhq_wild --dataset_path /path/to/afhq_wild_dataset --save_path segmentation/afhq_wild --mask --fp16
python generate_cams_dataset.py --dataset afhq_wild --dataset_path /path/to/afhq_wild_dataset --save_path segmentation/afhq_wild --mask --no-seg --fp16
python generate_cams_dataset.py --dataset afhq_wild --dataset_path /path/to/afhq_wild_dataset --save_path segmentation/afhq_wild --mask --no-seg --flip --fp16


python generate_cams_dataset.py --dataset lsun_church --dataset_path /path/to/lsun_church_dataset --save_path segmentation/lsun_church --mask
python generate_cams_dataset.py --dataset lsun_church --dataset_path /path/to/lsun_church_dataset --save_path segmentation/lsun_church --mask --no-seg
python generate_cams_dataset.py --dataset lsun_church --dataset_path /path/to/lsun_church_dataset --save_path segmentation/lsun_church --mask --no-seg --flip
python generate_cams_dataset.py --dataset lsun_church --dataset_path /path/to/lsun_church_dataset --save_path segmentation/lsun_church --mask --fp16
python generate_cams_dataset.py --dataset lsun_church --dataset_path /path/to/lsun_church_dataset --save_path segmentation/lsun_church --mask --no-seg --fp16
python generate_cams_dataset.py --dataset lsun_church --dataset_path /path/to/lsun_church_dataset --save_path segmentation/lsun_church --mask --no-seg --flip --fp16


python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /path/to/lsun_bedroom --save_path segmentation/lsun_bedroom --mask
python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /path/to/lsun_bedroom --save_path segmentation/lsun_bedroom --mask --no-seg
python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /path/to/lsun_bedroom --save_path segmentation/lsun_bedroom --mask --no-seg --flip
python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /path/to/lsun_bedroom --save_path segmentation/lsun_bedroom --mask --fp16
python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /path/to/lsun_bedroom --save_path segmentation/lsun_bedroom --mask --no-seg --fp16
python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /path/to/lsun_bedroom --save_path segmentation/lsun_bedroom --mask --no-seg --flip --fp16
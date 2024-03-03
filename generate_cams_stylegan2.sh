#!/bin/sh

python generate_cams_dataset_stylegan2.py --dataset ffhq --network /path/to/stylegan2-ffhq-network --save_path segmentation_stylegan2/ffhq --mask
python generate_cams_dataset_stylegan2.py --dataset ffhq --network /path/to/stylegan2-ffhq-network --save_path segmentation_stylegan2/ffhq --mask --no-seg
python generate_cams_dataset_stylegan2.py --dataset ffhq --network /path/to/stylegan2-ffhq-network --save_path segmentation_stylegan2/ffhq --mask --no-seg --flip
python generate_cams_dataset_stylegan2.py --dataset ffhq --network /path/to/stylegan2-ffhq-network --save_path segmentation_stylegan2/ffhq --mask --fp16
python generate_cams_dataset_stylegan2.py --dataset ffhq --network /path/to/stylegan2-ffhq-network --save_path segmentation_stylegan2/ffhq --mask --no-seg --fp16
python generate_cams_dataset_stylegan2.py --dataset ffhq --network /path/to/stylegan2-ffhq-network --save_path segmentation_stylegan2/ffhq --mask --no-seg --flip --fp16

python generate_cams_dataset_stylegan2.py --dataset afhq_cat --network /path/to/stylegan2-afhq-cat-network --save_path segmentation_stylegan2/afhq_cat --mask
python generate_cams_dataset_stylegan2.py --dataset afhq_cat --network /path/to/stylegan2-afhq-cat-network --save_path segmentation_stylegan2/afhq_cat --mask --no-seg
python generate_cams_dataset_stylegan2.py --dataset afhq_cat --network /path/to/stylegan2-afhq-cat-network --save_path segmentation_stylegan2/afhq_cat --mask --no-seg --flip
python generate_cams_dataset_stylegan2.py --dataset afhq_cat --network /path/to/stylegan2-afhq-cat-network --save_path segmentation_stylegan2/afhq_cat --mask --fp16
python generate_cams_dataset_stylegan2.py --dataset afhq_cat --network /path/to/stylegan2-afhq-cat-network --save_path segmentation_stylegan2/afhq_cat --mask --no-seg --fp16
python generate_cams_dataset_stylegan2.py --dataset afhq_cat --network /path/to/stylegan2-afhq-cat-network --save_path segmentation_stylegan2/afhq_cat --mask --no-seg --flip --fp16
# 
# 
python generate_cams_dataset_stylegan2.py --dataset afhq_wild --network /path/to/stylegan2-afhq-wild-network --save_path segmentation_stylegan2/afhq_wild --mask
python generate_cams_dataset_stylegan2.py --dataset afhq_wild --network /path/to/stylegan2-afhq-wild-network --save_path segmentation_stylegan2/afhq_wild --mask --no-seg
python generate_cams_dataset_stylegan2.py --dataset afhq_wild --network /path/to/stylegan2-afhq-wild-network --save_path segmentation_stylegan2/afhq_wild --mask --no-seg --flip
python generate_cams_dataset_stylegan2.py --dataset afhq_wild --network /path/to/stylegan2-afhq-wild-network --save_path segmentation_stylegan2/afhq_wild --mask --fp16
python generate_cams_dataset_stylegan2.py --dataset afhq_wild --network /path/to/stylegan2-afhq-wild-network --save_path segmentation_stylegan2/afhq_wild --mask --no-seg --fp16
python generate_cams_dataset_stylegan2.py --dataset afhq_wild --network /path/to/stylegan2-afhq-wild-network --save_path segmentation_stylegan2/afhq_wild --mask --no-seg --flip --fp16
# 

python generate_cams_dataset_stylegan2.py --dataset lsun_church --network /path/to/stylegan2-lsun-church-network --save_path segmentation_stylegan2/lsun_church --mask
python generate_cams_dataset_stylegan2.py --dataset lsun_church --network /path/to/stylegan2-lsun-church-network --save_path segmentation_stylegan2/lsun_church --mask --no-seg
python generate_cams_dataset_stylegan2.py --dataset lsun_church --network /path/to/stylegan2-lsun-church-network --save_path segmentation_stylegan2/lsun_church --mask --no-seg --flip
python generate_cams_dataset_stylegan2.py --dataset lsun_church --network /path/to/stylegan2-lsun-church-network --save_path segmentation_stylegan2/lsun_church --mask --fp16
python generate_cams_dataset_stylegan2.py --dataset lsun_church --network /path/to/stylegan2-lsun-church-network --save_path segmentation_stylegan2/lsun_church --mask --no-seg --fp16
python generate_cams_dataset_stylegan2.py --dataset lsun_church --network /path/to/stylegan2-lsun-church-network --save_path segmentation_stylegan2/lsun_church --mask --no-seg --flip --fp16

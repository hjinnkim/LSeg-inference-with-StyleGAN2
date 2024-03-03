#!/bin/sh

python generate_cams_dataset_stylegan2.py --dataset ffhq --network /home/khj/refactor/lang-seg/networks/ffhq-res256-mirror-paper256-noaug.pkl --save_path segmentation_stylegan2/ffhq --mask
python generate_cams_dataset_stylegan2.py --dataset ffhq --network /home/khj/refactor/lang-seg/networks/ffhq-res256-mirror-paper256-noaug.pkl --save_path segmentation_stylegan2/ffhq --mask --no-seg
python generate_cams_dataset_stylegan2.py --dataset ffhq --network /home/khj/refactor/lang-seg/networks/ffhq-res256-mirror-paper256-noaug.pkl --save_path segmentation_stylegan2/ffhq --mask --no-seg --flip
python generate_cams_dataset_stylegan2.py --dataset ffhq --network /home/khj/refactor/lang-seg/networks/ffhq-res256-mirror-paper256-noaug.pkl --save_path segmentation_stylegan2/ffhq --mask --fp16
python generate_cams_dataset_stylegan2.py --dataset ffhq --network /home/khj/refactor/lang-seg/networks/ffhq-res256-mirror-paper256-noaug.pkl --save_path segmentation_stylegan2/ffhq --mask --no-seg --fp16
python generate_cams_dataset_stylegan2.py --dataset ffhq --network /home/khj/refactor/lang-seg/networks/ffhq-res256-mirror-paper256-noaug.pkl --save_path segmentation_stylegan2/ffhq --mask --no-seg --flip --fp16

python generate_cams_dataset_stylegan2.py --dataset afhq_cat --network /home/khj/refactor/lang-seg/networks/stylegan2-ada-afhq-cat-config-f.pkl --save_path segmentation_stylegan2/afhq_cat --mask
python generate_cams_dataset_stylegan2.py --dataset afhq_cat --network /home/khj/refactor/lang-seg/networks/stylegan2-ada-afhq-cat-config-f.pkl --save_path segmentation_stylegan2/afhq_cat --mask --no-seg
python generate_cams_dataset_stylegan2.py --dataset afhq_cat --network /home/khj/refactor/lang-seg/networks/stylegan2-ada-afhq-cat-config-f.pkl --save_path segmentation_stylegan2/afhq_cat --mask --no-seg --flip
python generate_cams_dataset_stylegan2.py --dataset afhq_cat --network /home/khj/refactor/lang-seg/networks/stylegan2-ada-afhq-cat-config-f.pkl --save_path segmentation_stylegan2/afhq_cat --mask --fp16
python generate_cams_dataset_stylegan2.py --dataset afhq_cat --network /home/khj/refactor/lang-seg/networks/stylegan2-ada-afhq-cat-config-f.pkl --save_path segmentation_stylegan2/afhq_cat --mask --no-seg --fp16
python generate_cams_dataset_stylegan2.py --dataset afhq_cat --network /home/khj/refactor/lang-seg/networks/stylegan2-ada-afhq-cat-config-f.pkl --save_path segmentation_stylegan2/afhq_cat --mask --no-seg --flip --fp16
# 
# 
python generate_cams_dataset_stylegan2.py --dataset afhq_wild --network /home/khj/refactor/lang-seg/networks/stylegan2-ada-afhq-wild-config-f.pkl --save_path segmentation_stylegan2/afhq_wild --mask
python generate_cams_dataset_stylegan2.py --dataset afhq_wild --network /home/khj/refactor/lang-seg/networks/stylegan2-ada-afhq-wild-config-f.pkl --save_path segmentation_stylegan2/afhq_wild --mask --no-seg
python generate_cams_dataset_stylegan2.py --dataset afhq_wild --network /home/khj/refactor/lang-seg/networks/stylegan2-ada-afhq-wild-config-f.pkl --save_path segmentation_stylegan2/afhq_wild --mask --no-seg --flip
python generate_cams_dataset_stylegan2.py --dataset afhq_wild --network /home/khj/refactor/lang-seg/networks/stylegan2-ada-afhq-wild-config-f.pkl --save_path segmentation_stylegan2/afhq_wild --mask --fp16
python generate_cams_dataset_stylegan2.py --dataset afhq_wild --network /home/khj/refactor/lang-seg/networks/stylegan2-ada-afhq-wild-config-f.pkl --save_path segmentation_stylegan2/afhq_wild --mask --no-seg --fp16
python generate_cams_dataset_stylegan2.py --dataset afhq_wild --network /home/khj/refactor/lang-seg/networks/stylegan2-ada-afhq-wild-config-f.pkl --save_path segmentation_stylegan2/afhq_wild --mask --no-seg --flip --fp16
# 

python generate_cams_dataset_stylegan2.py --dataset lsun_church --network /home/khj/refactor/lang-seg/networks/stylegan2-church-config-f.pkl --save_path segmentation_stylegan2/lsun_church --mask
python generate_cams_dataset_stylegan2.py --dataset lsun_church --network /home/khj/refactor/lang-seg/networks/stylegan2-church-config-f.pkl --save_path segmentation_stylegan2/lsun_church --mask --no-seg
python generate_cams_dataset_stylegan2.py --dataset lsun_church --network /home/khj/refactor/lang-seg/networks/stylegan2-church-config-f.pkl --save_path segmentation_stylegan2/lsun_church --mask --no-seg --flip
python generate_cams_dataset_stylegan2.py --dataset lsun_church --network /home/khj/refactor/lang-seg/networks/stylegan2-church-config-f.pkl --save_path segmentation_stylegan2/lsun_church --mask --fp16
python generate_cams_dataset_stylegan2.py --dataset lsun_church --network /home/khj/refactor/lang-seg/networks/stylegan2-church-config-f.pkl --save_path segmentation_stylegan2/lsun_church --mask --no-seg --fp16
python generate_cams_dataset_stylegan2.py --dataset lsun_church --network /home/khj/refactor/lang-seg/networks/stylegan2-church-config-f.pkl --save_path segmentation_stylegan2/lsun_church --mask --no-seg --flip --fp16


# python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom --mask
# python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom --mask --no-seg
# python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom --mask --no-seg --flip
# python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom --mask --fp16
# python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom --mask --no-seg --fp16
# python generate_cams_dataset.py --dataset lsun_bedroom --dataset_path /media/khj/T7/datasets/converted/lsunbedroom256x256centercrop3m.zip --save_path segmentation/lsun_bedroom --mask --no-seg --flip --fp16

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
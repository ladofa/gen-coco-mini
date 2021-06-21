import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--coco_src_path', default='c:/dataset/coco')
parser.add_argument('--coco_dst_path', default='c:/dataset/coco_mini')
args = parser.parse_args()

import json
import os
import random
from collections import defaultdict
from shutil import copyfile

def get_mini(data, rate=0.01):
    data_dst = {}
    image_to_anno = defaultdict(list)
    for anno in data['annotations']:
        image_to_anno[anno['image_id']].append(anno)

    dst_images = []
    dst_annos = []
    for image in data['images']:
        if random.random() < rate:
            image_id = image['id']
            dst_images.append(image)
            dst_annos.extend(image_to_anno[image_id])

    data_dst['images'] = dst_images
    data_dst['annotations'] = dst_annos
    data_dst['categories'] = data['categories']
    return data_dst

train_json_src_path = os.path.join(args.coco_src_path, 'annotations', 'instances_train2017.json')
train_json_dst_path = os.path.join(args.coco_dst_path, 'annotations', 'instances_train2017.json')
val_json_src_path = os.path.join(args.coco_src_path, 'annotations', 'instances_val2017.json')
val_json_dst_path = os.path.join(args.coco_dst_path, 'annotations', 'instances_val2017.json')

train_data = json.load(open(train_json_src_path, 'r'))
val_data = json.load(open(val_json_src_path, 'r'))

train_data_dst = get_mini(train_data)
val_data_dst = get_mini(val_data, rate=0.05)

print(len(train_data['images']), len(train_data['annotations']))
print(len(train_data_dst['images']), len(train_data_dst['annotations']))
print(len(val_data['images']), len(val_data['annotations']))
print(len(val_data_dst['images']), len(val_data_dst['annotations']))

os.makedirs(args.coco_dst_path, exist_ok=True)
os.makedirs(os.path.join(args.coco_dst_path, 'train2017'), exist_ok=True)
os.makedirs(os.path.join(args.coco_dst_path, 'val2017'), exist_ok=True)
os.makedirs(os.path.join(args.coco_dst_path, 'annotations'), exist_ok=True)

json.dump(train_data_dst, open(train_json_dst_path, 'w'))
json.dump(val_data_dst, open(val_json_dst_path, 'w'))


def copy_images(images, src_dir, dst_dir):
    for image in images:
        file_name = image['file_name']
        image_path = os.path.join(src_dir, file_name)
        image_dst_path = os.path.join(dst_dir, file_name)
        copyfile(image_path, image_dst_path)

copy_images(
    train_data_dst['images'],
    os.path.join(args.coco_src_path, 'train2017'),
    os.path.join(args.coco_dst_path, 'train2017')
)

copy_images(
    val_data_dst['images'],
    os.path.join(args.coco_src_path, 'val2017'),
    os.path.join(args.coco_dst_path, 'val2017')
)

print('done.')
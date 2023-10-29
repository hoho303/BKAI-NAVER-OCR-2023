import numpy as np
import cv2
import os
import glob
import json 
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'raw', type=str, help='Raw images.'
    )
    parser.add_argument(
        'polygons', type=str, help='Polygons'
    )
    parser.add_argument(
        'cropped', type=str, help='Path to folder to contain cropped images'
    )

    args = parser.parse_args()
    return args

# Crop bbox
def crop_image(img , polygon):
    ## (1) Crop the bounding rect
    polygon = np.array(polygon)
    rect = cv2.boundingRect(polygon)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    ## (2) make mask
    polygon = polygon - polygon.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [polygon], -1, (255, 255, 255), -1, cv2.LINE_AA)
    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    return dst

# Calculate the area of the polygon
def find_area(polygon):
    polygon = np.array(polygon)
    x = polygon[:, 0]
    y = polygon[:, 1]

    S1 = np.sum(x*np.roll(y,-1))
    S2 = np.sum(y*np.roll(x,-1))

    area = .5*np.absolute(S1 - S2)

    return area

def relu(x):
    return x if x >= 0 else 0

def convert_polygon(polygon):
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    print('**** OLD POLYGON **** ')
    print(x_min, x_max, y_min, y_max)

    ratio = 0.3
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    x_min = relu(round(x_min - ratio * bbox_w))
    x_max = relu(round(x_max + ratio * bbox_w))
    y_min = relu(round(y_min - ratio * bbox_h))
    y_max = relu(round(y_max + ratio * bbox_h))

    print('***** NEW POLYGON')
    print(x_min, x_max, y_min, y_max)

    return [[x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]]

def filter_high_bboxes(bboxes, thresh):
    fil_bboxes = [bbox for bbox in bboxes if bbox[-1] >= thresh]
    return fil_bboxes

args = parse_args()
count_crop = 0
count_remain = 0
threshold = 0.5

mapping_ext = {}
for fname in os.listdir(args.raw):
    mapping_ext[fname.split('.')[0]] = fname

for path in glob.glob(os.path.join(args.polygons, "*.json")):
    with open(path) as f:
        content = json.load(f)
    
    bboxes = content['boundary_result']
    fil_bboxes = filter_high_bboxes(bboxes, threshold)
    img_name = os.path.basename(path).split('_', maxsplit=1)[1].split('.')[0]
    img_name_ext = mapping_ext[img_name]
    count = 0
    img = cv2.imread(f'{args.raw}/{img_name_ext}')
    cropped_path = f'{args.cropped}/{img_name_ext}'

    if len(fil_bboxes) != 1: 
        count_remain += 1
        cv2.imwrite(cropped_path, img)
    else:
        count_crop += 1
        bbox = fil_bboxes[0]
        score = bbox[-1]
        polygon = convert_polygon(bbox[:-1])      
                
        # Crop image
        cropped_img = crop_image(img, polygon)

        cv2.imwrite(cropped_path, cropped_img)

print(f'Remain  : {count_remain}')
print(f'Crop    : {count_crop}')


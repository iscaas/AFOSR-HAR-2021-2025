import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import defaultdict
import json
import cv2
import os
from natsort import natsorted

color = {
    'Nothing': (0, 0, 0),
    'Person': (0, 255, 0),
    'Car': (255, 0, 0),
    'Vehicle': (0, 255, 255),
    'Carrying_object': (0, 0, 255),
    'Bike/Bicycle': (255, 255, 0)
    }

# getting annotations
image_ids_annotations = defaultdict(list)

file = open('dataset/Large_Virat_Dataset/annotations/json_annotations/test_annotations.json')
anns = json.load(file)

# adding to datastructure
for ann in anns['annotations']:
    image_id = ann['image_id']
    image_ids_annotations[image_id].append(ann)
    

# getting mapping caregory_id to category name
category_id_to_name = dict()
for ann in anns['categories']:
    category_id_to_name[ann['id']] = ann['name']
    

keys = list(image_ids_annotations.keys())

directory_path = 'dataset/Large_Virat_Dataset/data/test/'
# Get all file names in the directory
image_files = natsorted(os.listdir(directory_path))

for i in range (len(image_files)):
    image_path = os.path.join(directory_path, image_files[i])
    image = cv2.imread(image_path)
    image_id = int(image_files[i].split(".")[0])
    image_anns = image_ids_annotations[keys[image_id]]
    
    text = "Frame #: "+str(i+1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_color = (0, 0, 0)  
    background_color = (70, 160, 210)

    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_width, text_height = text_size
    
    text_org = (0, 20)
    box_coord1 = (text_org[0], text_org[1] - text_height - 10)
    box_coord2 = (text_org[0] + text_width + 10, text_org[1] + 10)
    
    
    for image_ann in image_anns:
        bbox = image_ann['bbox']
        name = category_id_to_name[image_ann['category_id']]
        
        x = float(bbox[0])
        y = float(bbox[1])
        w = float(bbox[2])
        h = float(bbox[3])
        
        if name != 'Nothing':
            cv2.rectangle(image, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), color[name], 2)
            cv2.putText(image, name, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.rectangle(image, box_coord1, box_coord2, background_color, cv2.FILLED)
            cv2.putText(image, text, text_org, font, font_scale, text_color, font_thickness)
        else:
            cv2.rectangle(image, box_coord1, box_coord2, background_color, cv2.FILLED)
            cv2.putText(image, text, text_org, font, font_scale, text_color, font_thickness)
    
    cv2.imshow('image', image)
    key = cv2.waitKey(10)
    
    if key == ord('q'):
        break
    
cv2.destroyAllWindows()

import os
import json
def extract_person_data(data_path, file_name):
    
    if 'train' in file_name: t = 'train'
    else: t = 'val'
    new_file_path = f"{data_path}instances_{t}_person_only.json"

    if os.path.exists(new_file_path): 
        print(new_file_path, 'already exists.')
        return

    with open(data_path + file_name, 'r') as f:
        coco = json.load(f)

    person_id = next(cat['id'] for cat in coco['categories'] if cat['name'] == 'person')

    filtered_anns = [ann for ann in coco['annotations'] if ann['category_id'] == person_id]

    valid_image_ids = set(ann['image_id'] for ann in filtered_anns)
    filtered_images = [img for img in coco['images'] if img['id'] in valid_image_ids]

    filtered_coco = {
        'images': filtered_images,
        'annotations': filtered_anns,
        'categories': [cat for cat in coco['categories'] if cat['id'] == person_id]
    }

    with open(new_file_path, 'w') as f:
        json.dump(filtered_coco, f)



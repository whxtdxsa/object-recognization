import os
import json
def extract_person_data(original_file, new_file):

    with open(original_file, 'r') as f:
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

    with open(new_file, 'w') as f:
        json.dump(filtered_coco, f)



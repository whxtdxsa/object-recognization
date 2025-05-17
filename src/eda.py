save_path = "./src/experiments/chart/"


def extract_annotations(anno):
    print('Info of annotations------------------')
    print(anno.keys())
    print(anno['images'][0])
    print(anno['annotations'][0])
    print(anno['categories'][0])

    print('-------------------------------------\n')

from collections import Counter
def get_category_counts(anno):
    category_counts = Counter()
    categories = {cat['id']: cat['name'] for cat in annotations['categories']}

    for ann in anno['annotations']:
        category_counts[categories[ann['category_id']]] += 1

    return category_counts


def get_img_sizes(anno): 
    img_sizes = []
    
    for img in anno['images']:
        img_sizes.append((img['width'], img['height']))

    return img_sizes


def get_bbox_areas(anno):
    bbox_areas = []
    for ann in anno['annotations']:
        bbox = ann['bbox']
        bbox_areas.append(bbox[2] * bbox[3])
        
    return bbox_areas


def get_bbox_counts(anno):
    bbox_counts = Counter()
    for ann in anno['annotations']:
        bbox_counts[ann['image_id']] += 1

    return list(bbox_counts.values())


import matplotlib.pyplot as plt
def plot_category_nums(category_counts):
    counts_all = sorted(category_counts.values(), reverse=True)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(counts_all)), counts_all)

    plt.title("Categroy Frequency Rank")
    plt.xlabel("Category Rank")
    plt.ylabel("Count")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path + "categroy_fr.png")

    # plot top 20
    topk = 20
    category_counts = category_counts.most_common(topk)
    labels, counts = zip(*category_counts) 

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts)

    plt.title("Categroy Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path + "categroy_dist.png")
    plt.clf()


def plot_img_sizes(img_sizes):
    widths, heights = zip(*img_sizes)

    plt.figure(figsize=(10, 5))
    plt.scatter(widths, heights, alpha=0.3)

    plt.title("Image Size Distribution")
    plt.xlabel("Width")
    plt.ylabel("Height")
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path + "image_size_dist.png")
    plt.clf()


def plot_bbox_areas(bbox_areas):

    plt.figure(figsize=(10, 5))
    plt.hist(bbox_areas, bins=20)

    plt.title("Bounding Box Area Distribution")
    plt.xlabel("Area")
    plt.ylabel("Count")
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path + "bbox_area.png")
    plt.clf()


def plot_bbox_counts(bbox_counts):

    plt.figure(figsize=(10, 5))
    plt.hist(bbox_counts, bins=20)

    plt.title("Number of Objects per Image")
    plt.xlabel("Objects")
    plt.ylabel("Images")
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path + "object_per_img.png")
    plt.clf()


from collections import defaultdict
def get_image_id_dict(anno):
    image_id_to_info = {img['id']: img for img in anno['images']}

    image_id_to_bboxes = defaultdict(list)

    for ann in anno['annotations']:
        bbox = ann['bbox']
        x, y, w, h = bbox
        x_max = x + w
        y_max = y + h
        image_id_to_bboxes[ann['image_id']].append([x, y, x_max, y_max])

    return image_id_to_info, image_id_to_bboxes


from PIL import Image, ImageDraw
def show_bbox(image_path, bboxes):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:  # [x_min, y_min, x_max, y_max]
        draw.rectangle(bbox, outline="red", width=2)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path + "bbox.png", bbox_inches='tight', pad_inches=0)
    plt.clf()

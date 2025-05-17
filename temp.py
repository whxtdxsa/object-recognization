import json
img_path = "./data/images/val2017/"
label_path = "./data/labels/annotations/instances_val2017.json"

with open(label_path, "r") as f:
    data = json.load(f)

# 최상위 키 확인
# print(data.keys())
print(data['categories'][0])


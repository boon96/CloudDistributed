from numpy import append
from retinaface import RetinaFace
from generate_xml import GenerateXml

# Need to add some code to auto pull from images folder
def generate_xml(image_path):
    result = RetinaFace.detect_faces("images\{image_path}")
    bounding_boxes = []
    classes = []
    for item in result.values():
        print(item["facial_area"])
        bounding_boxes.append({'xmin': item[0] , 'xmax': item[1] , 'ymin': item[2], 'ymax': item[3]})
        classes.append("Face")

    xml = GenerateXml(bounding_boxes,image_width, image_height, classes, image_name)
    xml.gerenate_basic_structure()

# xml = GenerateXml([{'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}, {'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}, {'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}], '4000', '2000', ['miner', 'miner', 'rust'], 'Image_0')
# xml.gerenate_basic_structure()

# result = RetinaFace.detect_faces("img1.jpeg")
# for item in result.values():
#         print(item["facial_area"])

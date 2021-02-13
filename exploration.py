import json
import pprint
import shutil
from typing import Dict, Set, Tuple
from PIL import ImageDraw
from customImage import CustomImage

PERSON_CATEGORY_ID = 1


def load_json_file(path: str) -> Dict:
    with open(path) as file:
        return json.load(file)


def print_json_file(json_dict: Dict):
    pprint.pprint(json_dict)


def get_ids_of_images_with_people(json_dict: Dict, image_id_to_path: Dict[int, str]) -> Set[Tuple[int, str]]:
    image_ids = set()
    for annotation in json_dict['annotations']:
        if annotation['category_id'] == PERSON_CATEGORY_ID:
            image_id: int = annotation['id']
            file_name = image_id_to_path.get(image_id)
            if file_name:
                image_ids.add((image_id, file_name))
    return image_ids


def get_image_id_to_path_dict(json_dict) -> Dict[int, str]:
    images = json_dict['images']
    image_id_to_path_dict = dict()

    for image in images:
        image_id_to_path_dict[image['id']] = image['file_name']

    return image_id_to_path_dict


def copy_over_people_images_to_dest(id_string_set: Set[Tuple[int, str]]):
    ctr = 0
    for image_id, image_name in id_string_set:
        shutil.copy(f'./train2017/{image_name}', f'./people')
        ctr += 1
    print(f'Copied {ctr} images to destination.')


def gen_custom_image_dict(json_dict: Dict) -> Dict[int, CustomImage]:
    images = dict()
    for image in json_dict['images']:
        images[image['id']] = CustomImage(image_json=image)
    return images


def annotate_images(json_dict: Dict, custom_image_dict: Dict[int, CustomImage], people_ids: Set[Tuple[int, str]]):
    just_ids = {x[0] for x in people_ids}
    for annotation in json_dict['annotations']:
        image_id: int = annotation['id']
        if image_id not in just_ids:
            continue
        image = custom_image_dict.get(image_id)
        if image is None:
            continue
        image.draw_bounding_box(annotation)


def main():
    json_dict = load_json_file('./annotations_trainval2017/annotations/instances_train2017.json')
    image_id_to_path_dict = get_image_id_to_path_dict(json_dict)
    ids_of_people_images = get_ids_of_images_with_people(json_dict, image_id_to_path_dict)
    # print('\n'.join(list(map(str, get_ids_of_images_with_people(json_dict, image_id_to_path_dict)))))
    # copy_over_people_images_to_dest(ids_of_people_images)
    custom_image_dict = gen_custom_image_dict(json_dict)
    annotate_images(json_dict, custom_image_dict, ids_of_people_images)


if __name__ == '__main__':
    main()

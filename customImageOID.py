from PIL import Image, ImageDraw
from typing import Dict


class CustomImage:
    def __init__(self, image_json: Dict):
        self.file_name = image_json['file_name']
        self.id = image_json['id']
        self.annotated = False

    def draw_bounding_box(self, annotation):
        bbox = list(map(int, annotation['bbox']))
        top_left = tuple(bbox[:2])
        bottom_right = (top_left[0] + bbox[2], top_left[1] + bbox[3])

        base_image = Image.open(f'./people/{self.file_name}').convert('RGBA')
        new_image = Image.new('RGBA', base_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(new_image)
        draw.rectangle(xy=[top_left, bottom_right], width=3, outline=(153, 50, 204, 192))

        out = Image.alpha_composite(base_image, new_image).convert('RGB')
        out.save(f'./people_annotated/{self.id}.{self.file_name}')
        self.annotated = True


# if __name__ == '__main__':
#     image = CustomImage(
#         {
#             'file_name': './people/000000571263.jpg',
#             'id': '12345'
#         })
#     image.draw_bounding_box({'bbox': [50, 50, 50, 50]})

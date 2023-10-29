import os
import json
import random
from PIL import Image, ImageFont, ImageDraw, ImageChops
import math
from math import ceil, floor

# Đường dẫn đến các tệp và thư mục
font_path = "/home/visedit/WorkingSpace/BKAI2023/Datasets/SVN-Holiday.otf"
json_path = "full_text_trainingdata.json"
background_folder = '/home/visedit/WorkingSpace/BKAI2023/Datasets/backgrounds'
output_folder = '/home/visedit/WorkingSpace/BKAI2023/Datasets/Overlay'

# Tạo thư mục output nếu nó chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Đọc file JSON
with open(json_path, 'r', encoding='utf-8') as json_file:
    texts = json.load(json_file)

background_files = [f for f in os.listdir(background_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

total_images = 20000
center_straight_limit = total_images * 0.5
center_rotate_limit = total_images * 0.25
random_straight_limit = total_images * 0.25
random_rotate_limit = total_images * 0.25
padding_percentage = 0.05

center_straight_counter, center_rotate_counter, random_straight_counter, random_rotate_counter = 0, 0, 0, 0
bounding_boxes = {}
image_counter = 1

for idx, text in enumerate(texts):
    for background_file in background_files:
        with Image.open(os.path.join(background_folder, background_file)) as img:
            img = img.resize((int(img.width * 2), int(img.height * 2)), resample=Image.BILINEAR)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font_path, size=25)

            width, height = draw.textsize(text, font=font)

            # Set initial bounding box dimensions
            box_width, box_height = ceil(width), ceil(height)

            rotate_text = False
            position = None
            
            # Choose a random category for the image
            choices = ['random_straight', 'random_rotate']
            chosen = random.choice(choices)
            
            if chosen == 'random_rotate' and random_rotate_counter < random_rotate_limit:
                max_x = img.width - width
                max_y = img.height - height
                x = random.randint(0, abs(max_x))
                y = random.randint(0, abs(max_y))
                rotate_text = True
                random_rotate_counter += 1
            elif chosen == 'random_straight' and random_straight_counter < random_straight_limit:
                max_x = img.width - width
                max_y = img.height - height
                x = random.randint(0, abs(max_x))
                y = random.randint(0, abs(max_y))
                random_straight_counter += 1
            else:
                continue  

            if rotate_text:
                angle = random.choice([20, 340])
                angle_rad = math.radians(angle)

                # Calculate new dimensions after rotation
                box_width = int(width * abs(math.cos(angle_rad)) + height * abs(math.sin(angle_rad)))
                box_height = int(width * abs(math.sin(angle_rad)) + height * abs(math.cos(angle_rad)))

                # Create rotated text image
                padding = int(max(width, height) * 1.5)
                padded_img = Image.new('RGBA', (width + 2 * padding, height + 2 * padding), (255, 255, 255, 0))
                d = ImageDraw.Draw(padded_img)
                d.text((padding, padding), text, font=font, fill=(55, 48, 56))
                rotated_img = padded_img.rotate(-angle, expand=1)
                bbox = ImageChops.difference(rotated_img, Image.new('RGBA', rotated_img.size, (255, 255, 255, 0))).getbbox()
                cropped_img = rotated_img.crop(bbox)

                # Adjust position for rotated text
                x = x + (width - cropped_img.width) // 2
                y = y + (height - cropped_img.height) // 2
                img.paste(cropped_img, (int(x), int(y)), cropped_img)
            else:
                draw.text((x, y), text, font=font, fill=(55, 48, 56))

            # Save resultant image
            output_filename = f"{image_counter}.png"
            img.save(os.path.join(output_folder, output_filename))

            # Update bounding box information
            bounding_boxes[output_filename] = {
                'x': floor(x),
                'y': floor(y),
                'width': box_width,
                'height': box_height
            }

            image_counter += 1
            if image_counter > total_images:
                break

    if image_counter > total_images:
        break

# Save bounding box information to new JSON
with open('bounding_boxes.json', 'w', encoding='utf-8') as outfile:
    json.dump(bounding_boxes, outfile, ensure_ascii=False, indent=4)

print("Xử lý xong!")

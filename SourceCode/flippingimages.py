#Horizontaly flipped images for considering left-handed signers 

import os
import cv2

base_path = r'Sign-Language-Recognition-System\data\Alphabets\A'#path accordingly

# folder existance
if not os.path.exists(base_path):
    print(f"❌ Folder not found: {base_path}")
    exit()

# Get only .jpg images
images = [img for img in os.listdir(base_path) if img.endswith('.jpg')]

if not images:
    print(f"⚠️ No .jpg images found in: {base_path}")
    exit()

# Getting the last image number from original images (not flipped ones) for easier labeling
last_num = max([
    int(img.split('_')[1].split('.')[0]) 
    for img in images if img.startswith("Image_")
])

for idx, img_name in enumerate(images):
    if not img_name.startswith("Image_"):
        continue  # Skip flipped or other files

    img_path = os.path.join(base_path, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Can't read image: {img_path}")
        continue

    # Flip image horizontally 
    flipped_img = cv2.flip(img, 1)

    # Save with incremented filename
    new_img_num = last_num + idx + 1
    new_img_name = f'ImageFlipped_{new_img_num}.jpg'
    new_img_path = os.path.join(base_path, new_img_name)

    cv2.imwrite(new_img_path, flipped_img)

print("✅ Flipping complete for:", base_path)

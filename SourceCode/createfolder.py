import os
import string

base_path = "data"

for letter in string.ascii_uppercase:  # A to Z
    folder_path = os.path.join(base_path, letter)
    os.makedirs(folder_path, exist_ok=True)
    
for i in range(10):  # 0 to 9
    folder_path = os.path.join(base_path, str(i))
    os.makedirs(folder_path, exist_ok=True)
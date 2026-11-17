from pathlib import Path
from PIL import Image
dataset_path=Path(r"C:\Users\Ankush Agarwal\Downloads\archive\Cx22_Multi_Train")
image_dir=dataset_path/"Train"/"images"
mask_dir=dataset_path/"Train"/"gt_mask"/"cytoplasm"/"sem_seg"
print("image folder exist:",image_dir.exists())
print("mask folder exist:",mask_dir.exists())
image_files = list(image_dir.glob("*.png"))

mask_files = list(mask_dir.glob("*.png"))

print("Total original images:", len(image_files))

print("Total cytoplasm masks:", len(mask_files))
print("\nFirst 10 original image filenames:")

for file in image_files[:10]:
    print(file.name)


print("\nFirst 10 cytoplasm mask filenames:")

for file in mask_files[:10]:
    print(file.name)








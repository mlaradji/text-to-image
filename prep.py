### Preparation

# # Connect Drive
# from google.colab import drive

from pathlib import Path

# gdrive_dir = Path('/content/gdrive')
# drive.mount(str(gdrive_dir))

# Define data directory and make it if necessary
# base_dir = gdrive_dir / 'My Drive/Produvia/texttoimage'
#!rm -r /content/texttoimage
base_dir = Path(".")
base_dir.mkdir(parents=True, exist_ok=True)

main_dir = base_dir / ""
data_dir = main_dir / "data"
models_dir = main_dir / "models"
damsme_dir = main_dir / "DAMSMencoders"

data_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

# Download COCO dataset
from google_drive_downloader import GoogleDriveDownloader as gdd

files = [
    {"id": "1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ", "dir": data_dir / "birds.zip"},
    {"id": "1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9", "dir": data_dir / "coco.zip"},
    {"id": "1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V", "dir": damsme_dir / "bird.zip"},
    {"id": "1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ", "dir": damsme_dir / "coco.zip"},
    {
        "id": "1lqNG75suOuR_8gjoEPYNp8VyT_ufPPig",
        "dir": models_dir / "bird_AttnGAN2.pth",
    },
    {
        "id": "1i9Xkg9nU74RAvkcqKE-rJYhjvzKAMnCi",
        "dir": models_dir / "coco_AttnGAN2.pth",
    },
    {
        "id": "19TG0JUoXurxsmZLaJ82Yo6O0UJ6aDBpg",
        "dir": models_dir / "bird_AttnDCGAN2.pth",
    },
]

for file in files:
    if file["dir"].exists():
        print("Skipped downloading %s as it already exists." % file["dir"])
    else:
        gdd.download_file_from_google_drive(
            file_id=file["id"],
            dest_path=file["dir"],
            unzip=file["dir"].suffix == ".zip",
        )

# Change to the AttnGAN code directory.
# import os
# os.chdir(main_dir / 'code')

# Unzip files.
#!cd data/coco/ && unzip -qq val2014-text.zip && unzip -qq train2014-text.zip && ls

# Choose the dataset.
coco_dataset = "train2014"
while True:
    try:
        (data_dir / "coco" / "text").symlink_to(
            data_dir / "coco" / coco_dataset, target_is_directory=True
        )
        break
    except FileExistsError:
        (data_dir / "coco" / "text").unlink()
        continue

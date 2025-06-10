import os
import subprocess

# Define the directory paths
COCO_DIR = os.path.join(os.getenv("DATA_PATH", "./outputs/"), "refcoco_raw")
os.makedirs(COCO_DIR, exist_ok=True)
os.chdir(COCO_DIR)

# Define the URLs in a dictionary
urls = {
    "train2014.zip": "http://images.cocodataset.org/zips/train2014.zip",
    "refcoco.zip": "https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip",
    "refcoco+.zip": "https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip",
    "refcocog.zip": "https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
}

# Function to download and unzip files
def download_and_unzip(file_name, url, dest_dir):
    subprocess.run(["wget", url])
    subprocess.run(["unzip", file_name, "-d", dest_dir])
    os.remove(file_name)


# # Download and unzip the COCO dataset
download_and_unzip("train2014.zip", 
                   urls["train2014.zip"], 
                   "images/")

# List of ref data filenames
ref_data_files = ["refcoco.zip", 
                  "refcoco+.zip", 
                  "refcocog.zip"]

# Download and unzip the refcoco datasets
for file_name in ref_data_files:
    download_and_unzip(file_name, urls[file_name], ".")

print("All files downloaded and unzipped successfully.")

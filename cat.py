import os
import shutil
import tensorflow as tf

# Download the full cats vs dogs dataset (compressed)
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
path_to_zip = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=dataset_url, extract=True)

# Move it to current directory and rename
extracted_path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
if os.path.exists("cats_and_dogs"):
    shutil.rmtree("cats_and_dogs")
shutil.copytree(extracted_path + "/train", "cats_and_dogs")

print("✅ Sample image dataset is ready in ./cats_and_dogs/")

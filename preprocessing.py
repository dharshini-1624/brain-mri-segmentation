import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage import exposure
from skimage.util import img_as_ubyte
import os
from keras.preprocessing.image import ImageDataGenerator


def apply_clahe(image):
    if image.mode != 'L':
        image = image.convert('L') 
    image_array = np.array(image)
    
    
    clahe_image = exposure.equalize_adapthist(image_array, clip_limit=0.03)
    clahe_image = img_as_ubyte(clahe_image)
    
    return Image.fromarray(clahe_image)


def normalize_image(image):
    image_array = np.array(image, dtype=np.float32) / 255.0
    return Image.fromarray((image_array * 255).astype(np.uint8))


def load_data(data_dir):
    # Load all images from the specified directory
    all_images = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith(('.jpg', '.png', '.tif'))]
    
    mri_images = []
    mask_images = []
    
    for image_path in all_images:
        if '_mask' in image_path:  # Assumes masked images have "_mask" in the filename
            mask_images.append(image_path)
        else:
            mri_images.append(image_path)

    # Split the dataset into training and testing
    return train_test_split(mri_images, mask_images, test_size=0.2, random_state=42)



def preprocess_images(mri_paths, mask_paths):
    mri_processed = []
    mask_processed = []
    
    for mri_path, mask_path in zip(mri_paths, mask_paths):
        mri_img = Image.open(mri_path)
        mask_img = Image.open(mask_path)
        
        
        mri_img_clahe = apply_clahe(mri_img)
        mri_img_normalized = normalize_image(mri_img_clahe)
        
        
        mri_processed.append(np.array(mri_img_normalized))
        mask_processed.append(np.array(mask_img))
    
    return np.array(mri_processed), np.array(mask_processed)


def get_data_generators(X_train, y_train):
    data_gen_args = dict(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

  
    image_datagen.fit(X_train)
    mask_datagen.fit(y_train)

    
    return zip(image_datagen.flow(X_train, batch_size=32, seed=42),
               mask_datagen.flow(y_train, batch_size=32, seed=42))

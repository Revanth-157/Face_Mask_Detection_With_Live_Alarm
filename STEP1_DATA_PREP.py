import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

image_dir = "dataset/"
classes = ['mask', 'nomask']  # Folder names
image_list = []
label_list = []

# Load images and labels
for label, class_name in enumerate(classes):  # masked -> 0, unmasked -> 1
    class_path = os.path.join(image_dir, class_name)
    for filename in os.listdir(class_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                image_list.append(image)
                label_list.append(label)  # 0 for masked, 1 for unmasked
            else:
                print(f"Warning: {filename} could not be read and will be skipped.")

print(f"Total images loaded: {len(image_list)}")

# Preprocessing
def resize_images(images, target_size=(128, 128)):
    return np.array([cv2.resize(img, target_size) for img in images])

def grayscale_images(images):
    return np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images])

def normalize_images(images):
    return np.array([img / 255.0 for img in images])

# Apply preprocessing
image_list = resize_images(image_list)
image_list = grayscale_images(image_list)
image_list = normalize_images(image_list)
image_list = image_list[..., np.newaxis]  # Add channel dimension (128,128,1)

# Convert labels to numpy array
label_list = np.array(label_list)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    image_list, label_list, test_size=0.2, random_state=42, stratify=label_list
)

# Save preprocessed data
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")


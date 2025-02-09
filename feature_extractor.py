# import os
# import pickle
# actors = os.listdir('data')
# print(actors)
#
# filenames = []
#
# for actor in actors:
#     for file in os.listdir(os.path.join('data',actor)):
#         filenames.append(os.path.join('data',actor,file))
#
#
# print(filenames)
# print(len(filenames))
#
# pickle.dump(filenames,open('filenames.pkl','wb'))

from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
import pickle
import torch
from facenet_pytorch import MTCNN

# Initialize the InceptionResnetV1 model
mtcnn = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()


import gc  # Import garbage collector

def extract_features(image_path):
    # Load image using OpenCV
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return None

    # Check if image is too large
    height, width = img.shape[:2]
    if height > 1000 or width > 1000:  # Resize only if dimensions are larger than 1000x1000
        img = cv2.resize(img, (640, 480))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    box, prob = mtcnn.detect(img_rgb)

    if box is not None:
        x1, y1, x2, y2 = map(int, box[0])  # Get face bounding box
        face = img_rgb[y1:y2, x1:x2]  # Crop the detected face

        if face.size != 0:
            face_resized = cv2.resize(face, (160, 160))

            # Normalize image (required for InceptionResnetV1)
            face_resized = (face_resized / 255.0 - 0.5) * 2.0

            # Convert the face to a tensor
            face_tensor = torch.tensor(np.moveaxis(face_resized, -1, 0), dtype=torch.float).unsqueeze(0)

            # Extract features (embedding) using the model
            with torch.no_grad():
                embedding = model(face_tensor)

            # Release memory
            del img, img_rgb, face, face_resized, face_tensor
            gc.collect()  # Force garbage collection

            return embedding[0].cpu().numpy()
        else:
            print(f"No valid face detected in image: {image_path}")
            return None
    else:
        print(f"No face detected in image: {image_path}")
        return None



filenames = pickle.load(open('filenames.pkl', 'rb'))


# Extract features for all images and save them in a dictionary
def extract_all_features(filenames):
    features = {}

    for filename in filenames:
        embedding = extract_features(filename)  # Assuming extract_features is already defined
        features[filename] = embedding

    # Save the extracted features
    pickle.dump(features, open('all_features.pkl', 'wb'))
    print("Features extraction completed.")


# Run the feature extraction for all images
extract_all_features(filenames)

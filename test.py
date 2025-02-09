import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
import pickle
from scipy.spatial.distance import cosine



# Initialize the MTCNN (face detector) and InceptionResnetV1 (feature extractor)
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

with open('all_features.pkl', 'rb') as f:
    features = pickle.load(f)

# Function to extract features from an image after detecting the face
def extract_face_and_features(image_path):
    # Load image using OpenCV
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    boxes, probs = mtcnn.detect(img_rgb)

    if boxes is not None:
        # For simplicity, assume we're working with the first detected face (you can extend it for multiple faces)
        face = img_rgb[int(boxes[0][1]):int(boxes[0][3]), int(boxes[0][0]):int(boxes[0][2])]

        # cv2.imshow("Cropped Face", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))  # Convert back to BGR for OpenCV
        # cv2.waitKey(0)  # Wait until a key is pressed
        # cv2.destroyAllWindows()  # Close the window
        
        # Resize the face to 160x160 as required by InceptionResnetV1
        face_resized = cv2.resize(face, (160, 160))

        # Normalize image (required for InceptionResnetV1)
        face_rgb = (face_resized / 255.0 - 0.5) * 2.0

        # Convert the face to a tensor
        face_tensor = torch.tensor(np.moveaxis(face_rgb, -1, 0), dtype=torch.float).unsqueeze(0)

        # Extract features (embedding) using the model
        with torch.no_grad():
            embedding = model(face_tensor)

        # Return the feature vector (embedding)
        return embedding[0].cpu().numpy()
    else:
        print("No face detected.")
        return None


# Example usage:
image_path = 'Olsen.jpg'
embedding = extract_face_and_features(image_path)

# if embedding is not None:
#     print("Face features extracted:", embedding)
# else:
#     print("No face detected in the image.")

if embedding is None:
    print("Error: No embedding extracted. Face not detected.")
else:
    similarity = []

    # Compute similarity only if embeddings are valid
    for filename, stored_embedding in features.items():
        if stored_embedding is not None:  # Ensure stored embeddings are valid
            sim = 1 - cosine(embedding.flatten(), stored_embedding.flatten())  # Ensure 1-D arrays
            similarity.append((filename, sim))  # Store (filename, similarity)

    # Check if similarity list is not empty
    if similarity:
        # Find the best match
        best_match = max(similarity, key=lambda x: x[1])  # Find highest similarity
        best_match_filename = best_match[0]
        print(f"Best match: {best_match_filename} with similarity {best_match[1]:.2f}")

        # Display the best-matching image
        temp_img = cv2.imread(best_match_filename)
        cv2.imshow('Best Match', temp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No valid matches found.")

import os
import shutil
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
import torch
from flask import Flask, request, render_template, send_from_directory
from scipy.spatial.distance import cosine


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load MTCNN and FaceNet model
mtcnn = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load stored features
with open('all_features.pkl', 'rb') as f:
    features = pickle.load(f)


def extract_features(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    box, _ = mtcnn.detect(img_rgb)

    if box is not None:
        x1, y1, x2, y2 = map(int, box[0])
        face = img_rgb[y1:y2, x1:x2]
        face_resized = cv2.resize(face, (160, 160))
        face_rgb = (face_resized / 255.0 - 0.5) * 2.0
        face_tensor = torch.tensor(np.moveaxis(face_rgb, -1, 0), dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            embedding = model(face_tensor)
        return embedding[0].cpu().numpy()
    else:
        return None


def find_best_match(embedding):
    similarity = []
    for filename, stored_embedding in features.items():
        if stored_embedding is not None:  # Avoid NoneType errors
            sim = 1 - cosine(embedding.flatten(), stored_embedding.flatten())
            similarity.append((filename, sim))

    if similarity:
        best_match = max(similarity, key=lambda x: x[1])
        best_match_path = best_match[0]  # Return the path of the best match
        celebrity_name = os.path.basename(os.path.dirname(best_match[0]))
        return best_match_path,celebrity_name
    return None


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            embedding = extract_features(filepath)

            if embedding is not None:
                best_match_path,celebrity_name = find_best_match(embedding)

                if best_match_path:
                    best_match_filename = os.path.basename(best_match_path)
                    result_path = os.path.join(RESULT_FOLDER, best_match_filename)

                    # Copy the best match image to the results folder
                    shutil.copy(best_match_path, result_path)

                    return render_template('index.html', user_image=file.filename, match_image=best_match_filename,celebrity_name=celebrity_name)
                else:
                    return "No matching face found."
            else:
                return "No face detected. Try another image."

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)  # Ensure result images are accessible


if __name__ == '__main__':
    app.run(debug=True)

# 🔍 Face Recognition System using FaceNet + SVM

This project implements a face recognition system using the **FaceNet embedding model** and an **SVM classifier**. It performs face detection, feature extraction, and identity recognition from static images. Real-time support can be added optionally.

---

## 🚀 Features

- ✅ Face detection using Haar Cascade or MTCNN  
- ✅ Feature extraction using pre-trained **FaceNet** model (128D embeddings)  
- ✅ Classification using **SVM (Support Vector Machine)**  
- ✅ Support for multiple identities  
- ✅ Embeddings and model saving/loading  
- ✅ Tested in **Jupyter Notebook (`face.ipynb`)**

---

## 🧠 How It Works

1. **Face Detection**  
   - Detect faces in input images using Haar Cascade or MTCNN

2. **Feature Extraction**  
   - Extract 128D facial embeddings using FaceNet (`facenet_keras.h5`)

3. **Training**  
   - Train an SVM classifier on embeddings

4. **Recognition**  
   - Predict identity of unknown faces using trained SVM

---

## 📁 Project Structure

```
face-recognition/
│
├── face.ipynb              # Main notebook with full pipeline
├── Memory
   ├── Models
      ├── trained_model.yml       # Saved SVM model   # Indian_Celebrities_trained_model.yml,
                              SportsCelebraties_trained_model.yml
   ├── labels.npy              # Labels for training faces
   ├── features.npy            # 128D embeddings
├── Faces/data/             # Folder structure: Faces/data/person_name/image.jpg
└── README.md               # You're here!
```

---

## 🛠️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

**Or individually:**

```bash
pip install numpy opencv-python mtcnn keras tensorflow scikit-learn joblib
```

---

## 📦 Dataset Structure

Organize your dataset like:

```
Faces/data/
├── person1/
│   ├── image1.jpg
│   └── image2.jpg
├── person2/
│   └── image1.jpg
```

Each folder represents a different identity.

---

## 📊 Training & Evaluation

- Embeddings are extracted using FaceNet.
- Train/test split is used with **stratification** for balanced classes.
- An **SVM classifier** is trained and saved using `joblib`.

```python
from sklearn.svm import SVC
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)
```

---

## 🔮 Predicting New Faces

Use the trained model to recognize new images:

```python
embedding = get_embedding(face)
predicted_label = model.predict([embedding])
```

---

## ✅ To Do

- [ ] Add real-time recognition with webcam
- [ ] UI for image upload and testing
- [ ] Support for retraining with new data

---

## 📚 References

- [FaceNet Paper](https://arxiv.org/abs/1503.03832)
- [Facenet Keras GitHub](https://github.com/nyoki-mtl/keras-facenet)
- [MTCNN Face Detection](https://github.com/ipazc/mtcnn)

---

## 🧑‍💻 Author

Developed with ❤️ using OpenCV, Keras, and Scikit-learn.

---

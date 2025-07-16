import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import os
import cv2

dir_name = 'dataset/faces/'
images = []
labels = []
label_names = os.listdir(dir_name)

for idx, person_name in enumerate(label_names):
    person_dir = os.path.join(dir_name, person_name)
    if not os.path.isdir(person_dir):
        continue
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (100, 100))
        images.append(img_resized.flatten())
        labels.append(idx)

images = np.array(images)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.4, random_state=42)
mean_face = np.mean(X_train, axis=0)
X_train_zero = X_train - mean_face
X_test_zero = X_test - mean_face

k_values = [10, 20, 30, 40, 50, 60, 70, 80]
accuracies = []

for k in k_values:
    pca = PCA(n_components=k, whiten=True, svd_solver='randomized', random_state=42)
    X_train_pca = pca.fit_transform(X_train_zero)
    X_test_pca = pca.transform(X_test_zero)
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=1)
    clf.fit(X_train_pca, y_train)
    acc = np.mean(clf.predict(X_test_pca) == y_test) * 100
    accuracies.append(acc)
    print(f'k={k}: Accuracy={acc:.2f}%')

plt.plot(k_values, accuracies, marker='o')
plt.xlabel('Number of Principal Components (k)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs k')
plt.show()

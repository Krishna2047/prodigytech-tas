import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


data_dir = 'D:/house-prices-advanced-regression-techniques/PetImages'  


images = []
labels = []


for label in ['cat', 'dog']:
    folder = os.path.join(data_dir, label)
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
       
        img = cv2.imread(img_path)
        if img is not None:  
           
            img = cv2.resize(img, (64, 64))  
            images.append(img.flatten())  
            labels.append(0 if label == 'cat' else 1)  

X = np.array(images)
y = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(StandardScaler(), SVC(kernel='linear'))


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))
print(confusion_matrix(y_test, y_pred))

# OCR Entity Extraction with Machine Learning

This project aims to extract specific entities from images, such as weight, dimensions, and other product specifications, using Optical Character Recognition (OCR) with Tesseract and machine learning for refined predictions.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)

---

## Overview

This project focuses on extracting key information, such as `item_weight`, `length`, `width`, `depth`, `voltage`, `wattage`, and more from images containing text. By utilizing Tesseract OCR for text recognition and a machine learning model for entity extraction, this solution can improve efficiency in data entry and verification in sectors like e-commerce, healthcare, and manufacturing.

## Dataset

The dataset consists of images and their corresponding entity labels, such as:
- **item_weight**: e.g., "620 grams"
- **dimensions**: e.g., "length", "width", "depth"
- **voltage**: e.g., "220V"
- **Other Specifications**: for various product attributes

The images are located in a folder (e.g., `resized_train_new_images`), and the label information is stored in a CSV file (e.g., `labels.csv`), with columns specifying `entity_name` and `entity_value`.

## Requirements

Install the required packages:

```bash
pip install pytesseract opencv-python numpy pandas scikit-learn
```

## Project Structure
├── data/

│   ├── resized_train_new_images/   # Folder with training images

│   └── labels.csv                  # CSV file with entity labels

├── notebooks/

│   └── ocr_entity_extraction.ipynb # Notebook for processing and model training

├── models/

│   └── best_model.pkl              # Trained machine learning model for entity extraction

└── README.md

## Preprocessing
The preprocessing steps include:

Image Resizing: Standardizing image dimensions.

Grayscale Conversion: Converting images to grayscale for improved OCR accuracy.

Thresholding: Applying thresholding to enhance text visibility.

Text Extraction: Using Tesseract OCR to extract text from images.

```bash
import cv2
import pytesseract

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img

def extract_text(img_path):
    img = preprocess_image(img_path)
    text = pytesseract.image_to_string(img)
    return text
```
## Modeling
The project uses machine learning to classify and extract relevant entity values from OCR-extracted text. A model like Random Forest or Gradient Boosting was trained on the extracted text and associated entity labels.
```bash
from sklearn.ensemble import RandomForestClassifier

# Define and train model
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
```

## Evaluation
Evaluate the model using metrics such as accuracy, precision, recall, and F1-score:
```bash
from sklearn.metrics import classification_report

# Make predictions and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```
## Results
F1 Score: Achieved an F1 score of 0.622 on test data.

Precision & Recall: Precision and recall were improved with preprocessing steps and by focusing on relevant entity classes.

## Usage
1.Clone the repository:

```bash
git clone https://github.com/yourusername/ocr-entity-extraction.git
```
2.Install dependencies:

```bash
pip install -r requirements.txt
```
3.Run the notebook ocr_entity_extraction.ipynb to preprocess images, train the model, and evaluate its performance.

4.Use the trained model (models/best_model.pkl) to extract entities from new images.

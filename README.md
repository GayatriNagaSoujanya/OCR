# Entity Value Extraction from Images

This project focuses on extracting specific entity values from images, particularly for attributes like weight, dimensions, voltage, wattage, etc., in use cases such as healthcare, e-commerce, and content moderation. The goal is to predict values like "620 grams" or "12 volts" based on visual data and associate them with corresponding entity names.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [Model Training](#model-training)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The model is designed to extract entity values associated with attributes (e.g., item weight, voltage) from images. The project uses OCR (Optical Character Recognition) techniques and natural language processing to identify and extract these values for specified entities.

## Dataset
- **Training Dataset**: Contains images with labels for `entity_name` (e.g., "item_weight") and `entity_value` (e.g., "620 grams").
- **Test Dataset**: Contains images with only `entity_name`, and the model predicts the corresponding `entity_value`.
  
**Classes**: The model predicts values for the following attributes:
  - Item weight
  - Length
  - Width
  - Depth
  - Voltage
  - Wattage
  - Maximum weight recommendation
  - Item volume

## Setup and Installation

### Prerequisites
- Python 3.7+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- Jupyter Notebook (optional for exploration)

### Install Dependencies
```bash
pip install -r requirements.txt

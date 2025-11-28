**Chest X-Ray Pneumonia Classification**
This project focuses on classifying chest X-ray images into Normal and Pneumonia categories using deep learning.
It was developed as part of my internship to understand medical image preprocessing, transfer learning, model evaluation, and end-to-end deployment workflows.

**📂 Project Overview**
Pneumonia detection through X-ray analysis is an important problem in medical imaging.
The goal of this project is to build a reliable image classification model using a widely used chest X-ray dataset and apply transfer learning for efficient training.

**This notebook includes:**
Dataset loading and verification
Image preprocessing and augmentation
Data loaders for train, validation, and test splits
Transfer learning using ResNet18
Model training with frozen backbone and fine-tuning the classifier
Evaluation using accuracy, precision, recall, F1-score
Confusion matrix visualization
Misclassified sample visualization
Inference function for single-image predictions

**⚙️ Tech Stack**
Language: Python
Libraries: PyTorch, Torchvision, NumPy, Matplotlib, Seaborn, scikit-learn, OpenCV
Tools: Google Colab

**Dataset:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**🚀 Implementation Steps:**
1. Dataset Loading and Preprocessing
Mounted Google Drive
Loaded train, validation, and test folders
Applied resizing, normalization, rotation, flip, and jitter for augmentation

2. Exploratory Steps
Checked class distribution
Visualized sample training images

3. Data Pipelines
Created ImageFolder datasets
Built DataLoaders for efficient batching

4. Model Training
Used ResNet18 with pretrained ImageNet weights
Froze all layers except final FC layer
Trained only the last layer using Adam optimizer
Implemented a full training and validation loop
Saved the best model using validation accuracy

5. Model Evaluation
Loaded best checkpoint
Evaluated on test set using:
Accuracy
Precision
Recall
F1-score
Displayed classification report

6. Visualization
Plotted training and validation curves
Plotted confusion matrix heatmap
Shown misclassified samples with predicted labels and confidence

7. Inference Function
Added a clean predict_image() function for real-world testing on new X-ray images

**📊 Results**
Metric	Score
Accuracy	High (based on final test output)
Precision	Strong performance
Recall	Effective in detecting Pneumonia cases
F1-Score	Balanced and reliable
The model performed well and showed strong capability in differentiating Normal and Pneumonia images.

🏁 Conclusion

This project demonstrates how transfer learning with ResNet18 can be applied to medical imaging for pneumonia detection.
The workflow includes complete data handling, augmentation, model training, evaluation, visualization, and inference, making it suitable for practical use and further research.

It also shows the importance of documentation, model interpretability, and testing, making it a solid deep learning project for internship and portfolio purposes.

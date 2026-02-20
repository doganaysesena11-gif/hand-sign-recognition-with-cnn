# Hand Sign Recognition with CNN

This project is about **recognizing American Sign Language (ASL) letters** from images of hand signs. The goal is to train a computer to **identify letters based on hand gestures**.

---

## Overview

- The project uses a **Convolutional Neural Network (CNN)**, a type of deep learning model designed for image analysis.  
- The model is trained on grayscale images of hands forming letters, then tested on unseen images to evaluate performance.  

---

## Dataset

The dataset used is **Sign Language MNIST**, available on Kaggle:  

[Sign Language MNIST Dataset](https://www.kaggle.com/datasets/ayesenadoan/sign-language-mnist)  

- **Training set:** `sign_mnist_train.csv`  
- **Test set:** `sign_mnist_test.csv`  
- Each image is **28x28 pixels**, grayscale.  
- Labels correspond to **letters A–Y (excluding J)**.  

---

## Methodology

1. **Data Preparation**  
   - Images are normalized to the range 0–1.  
   - Labels are converted to **one-hot encoding** for multi-class classification.  

2. **Model Architecture**  
   - **Convolutional layers** extract spatial features from images.  
   - **Pooling layers** reduce dimensionality while preserving important patterns.  
   - **Dense layers** interpret features for classification.  
   - **Dropout** is used to prevent overfitting.  

3. **Training**  
   - The model is trained on the training set with **categorical crossentropy loss** and **Adam optimizer**.  
   - A portion of the training set is used for validation to monitor performance.  

4. **Evaluation**  
   - After training, the model is evaluated on the test set to measure generalization.  

5. **Visualization**  
   - Sample images are displayed with labels to show the dataset and the types of hand signs used.  
   - Misclassified examples can be visualized to identify areas for improvement.  

---

## Results

- Test accuracy is approximately **90%**.  
- Visualization helps understand both the data and the model’s predictions.  

---

## Future Work

- Apply **data augmentation** to make the model more robust to variations in hand orientation, size, or lighting.  
- Explore deployment for **real-time hand sign recognition** using a camera.  
- Experiment with more advanced CNN architectures or **transfer learning** to improve accuracy.  

---

## References

- Dataset: [Sign Language MNIST on Kaggle](https://www.kaggle.com/datasets/ayesenadoan/sign-language-mnist)  
- TensorFlow/Keras CNN tutorials  


# Brain MRI Metastasis Segmentation

This repository contains implementations of two advanced segmentation architectures, Nested U-Net (U-Net++) and Attention U-Net, tailored for the task of segmenting brain metastases in MRI images. The project includes data preprocessing, model training, evaluation, and a user-friendly interface using Streamlit.

## 1. Model Architectures

### Nested U-Net (U-Net++)
Nested U-Net, or U-Net++, enhances the traditional U-Net architecture by adding nested skip pathways between the encoder and decoder. This structure allows the model to leverage features at various scales, bridging the semantic gap that often hinders segmentation accuracy.

In the context of metastasis segmentation:
- The nested architecture enables the model to better capture the complex shapes and features of metastases in the brain.
- Improved feature reuse from earlier layers allows for more precise localization of tumors, which is critical for accurate segmentation.

### Attention U-Net
Attention U-Net integrates attention mechanisms into the U-Net architecture. It enables the model to focus on the most relevant features while disregarding background noise. This is particularly useful in medical images where tumors may appear in varied contexts.

In metastasis segmentation:
- Attention mechanisms allow the model to highlight regions that are more likely to contain metastases, improving segmentation accuracy and reducing false positives.
- The model dynamically adjusts its focus based on the input, enhancing performance in challenging cases with complex backgrounds.

## 2. Data Preparation

### Combined Images
The dataset consists of combined MRI images containing both masked and unmasked regions. The data preprocessing involves:
- **Contrast Limited Adaptive Histogram Equalization (CLAHE)** to enhance the visibility of metastases.
- **Normalization** to scale the pixel values.
- **Data Augmentation** to increase the robustness of the models and prevent overfitting by applying transformations such as rotation, translation, and flipping.

### Data Loading
The data loading function has been modified to handle combined images. The logic extracts both the MRI images and their corresponding masks, allowing for effective training and evaluation of the segmentation models.

## 3. Instructions for Setup and Running

### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- FastAPI
- Streamlit
- Other required libraries (listed in `requirements.txt`)

### Installation Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Organize your MRI images and masks in a specified directory as shown below:
     ```
     /path_to_dataset/
     ├── combined_images/
     ```

4. Set the paths to your combined images in `train_evaluate.py`.

5. Train the models:
   ```bash
   python train_evaluate.py
   ```

6. Run the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```

7. Run the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```

## 4. Challenges in Brain Metastasis Segmentation

### Specific Challenges
- **Variability in Tumor Appearance**: Brain metastases can vary significantly in shape, size, and texture, making it difficult for models to generalize well across different cases.
- **Noise and Artifacts**: MRI images often contain noise and artifacts that can mislead the model, resulting in inaccurate segmentations.
- **Class Imbalance**: In many datasets, the number of pixels representing metastases is much smaller than those representing normal tissue, leading to class imbalance issues.

### Implementation Solutions
- **U-Net++**: By utilizing a nested architecture, U-Net++ captures features at multiple levels, helping to improve segmentation accuracy across varying tumor appearances.
- **Attention Mechanisms**: Attention U-Net focuses on the relevant features in the images, reducing the impact of noise and improving the model's ability to distinguish between normal and metastatic tissues.
- **Data Augmentation**: Implementing data augmentation strategies helps mitigate overfitting and improves the model's robustness against variability in the dataset.

## 5. Acknowledgments

- [TensorFlow](https://www.tensorflow.org/) for providing the deep learning framework.
- [Keras](https://keras.io/) for the high-level neural networks API.
- [Streamlit](https://streamlit.io/) for making it easy to create web apps for machine learning projects.



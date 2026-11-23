# ASL Hand Gesture Recognition System

A real-time American Sign Language (ASL) fingerspelling recognition system developed using Python, OpenCV, MediaPipe, and Machine Learning.

The system uses hand landmark detection and a Random Forest classifier to recognize static ASL alphabet gestures through a webcam and convert recognized letters into text.

## Project Overview

Communication is essential for everyone, but people with hearing or speech impairments may face communication barriers. This project aims to help bridge that gap by recognizing American Sign Language (ASL) hand gestures and converting them into readable text.

The application captures live webcam input, detects hand landmarks, extracts relevant features, and uses a trained machine learning model to predict the corresponding ASL letter.

**Note:** This implementation focuses on static ASL alphabet gestures. Letters J and Z require movement-based gestures and are excluded. Therefore, the system recognizes 24 static letters: A–I and K–Y.

## Features

- Real-time hand detection using a webcam.
- Hand landmark extraction using MediaPipe.
- Feature normalization for improved model input.
- ASL alphabet classification using Random Forest.
- Custom dataset collection for model training.
- Sentence construction from consecutive letter predictions.
- Prediction stability checking to reduce incorrect detections.
- Display of recognized letters and constructed text.
- Model evaluation using a confusion matrix.
- Dataset preprocessing and normalization.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Webcam access and image processing |
| MediaPipe | Hand landmark detection |
| Scikit-learn | Random Forest classification and model evaluation |
| NumPy | Numerical operations |
| Pandas | Dataset handling and preprocessing |
| Joblib | Saving and loading the trained model |

## Project Structure

```text
asl_project/
│
├── data/
│   ├── data.csv
│   └── data_normalized.csv
│
├── models/
│   └── asl_letter_model.pkl
│
├── gestures/
│   └── (Optional gesture template files)
│
├── infer.py
├── collect_data.py
├── landmarks.py
├── train.py
├── migrate_data.py
├── analyze_confusion.py
├── sentence_builder.py
├── remove_letter.py
├── clean_sentence.py
│
├── .gitignore
└── README.md
```

*The folder structure may vary depending on the files included in the repository.*

## How It Works

The system follows these main stages:

### 1. Webcam Input

OpenCV captures live video from the computer's webcam.

### 2. Hand Detection

MediaPipe detects the hand and identifies its 21 landmarks.

Each landmark contains three coordinates (X, Y, Z), giving 63 raw coordinate values.

### 3. Feature Extraction and Normalization

The extracted landmarks are normalized to make the features more consistent across different hand positions and sizes.

Additional features, including fingertip distances and finger curl angles, are calculated.

The resulting feature vector contains 72 values.

### 4. Machine Learning Classification

A trained Random Forest classifier receives the normalized feature vector and predicts the corresponding ASL alphabet letter.

### 5. Sentence Construction

The SentenceBuilder component processes consecutive predictions, checks prediction stability, and builds readable text.

A pause without a detected hand can be used to separate words.

### 6. Output Display

OpenCV displays the live webcam feed, recognized letter, and constructed text.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version supported by the installed dependencies.
- Git
- VS Code (recommended)
- A working webcam

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/asl-recognition.git
```

Navigate into the project directory:

```bash
cd asl-recognition
```

### Step 2: Create a Virtual Environment

```bash
python -m venv asl-env
```

Activate it on Windows:

```powershell
asl-env\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn joblib
```

If the project includes a `requirements.txt`, install the dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Project

Follow these steps in order.

### Step 1: Collect Training Data

Open `collect_data.py` and set the alphabet letter you want to record.

For example:

```python
LETTER = "A"
```

Run:

```bash
python collect_data.py
```

Repeat the collection process for all 24 static ASL letters:

**A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y**

Collect multiple samples for each letter to improve model performance.

The collected raw landmark data is stored in:

```text
data/data.csv
```

### Step 2: Normalize the Dataset

Run:

```bash
python migrate_data.py
```

This processes the raw landmark coordinates and generates the normalized dataset:

```text
data/data_normalized.csv
```

### Step 3: Train the Model

Run:

```bash
python train.py
```

The training script uses a Random Forest classifier to learn the relationship between normalized hand features and their corresponding ASL letters.

The trained model is saved as:

```text
models/asl_letter_model.pkl
```

### Step 4: Evaluate the Model

Run:

```bash
python analyze_confusion.py
```

This script evaluates classification performance and displays a confusion matrix to help identify letters that the model frequently confuses.

### Step 5: Start Real-Time Recognition

Run:

```bash
python infer.py
```

The webcam will open and the application will begin recognizing static ASL hand gestures.

**Keyboard Controls:**

| Key | Function |
|---|---|
| Q | Quit the application |
| C | Clear the constructed sentence |

## Machine Learning Model

The project uses the Random Forest algorithm for multi-class classification.

Random Forest is an ensemble learning algorithm that combines multiple decision trees to produce a final prediction.

The model is configured with parameters such as:

- Number of estimators: 400
- Maximum tree depth: 20
- Minimum samples per leaf: 2
- Class weighting: Balanced

These parameters are defined in the training script and can be adjusted based on experimental results.

## Dataset

The dataset consists of hand landmark features collected through a webcam.

Each sample contains:

- 63 raw landmark coordinate values.
- 9 additional normalized geometric features.
- One corresponding alphabet label.

The normalized dataset is used to train and evaluate the machine learning model.

## Limitations

- Recognizes 24 static ASL alphabet gestures.
- Dynamic gestures for J and Z are not supported in this static-only implementation.
- Recognition accuracy may vary depending on lighting, camera quality, hand orientation, and background conditions.
- The model's performance depends on the quality and diversity of the collected training data.
- This is an educational prototype and is not a complete ASL translation system.

## Future Enhancements

- Support for dynamic ASL gestures, including J and Z.
- Recognition of complete ASL words and sentences.
- Improved recognition accuracy through larger and more diverse datasets.
- Support for two-hand gestures.
- Text-to-speech output for recognized sentences.
- A graphical user interface for easier interaction.
- Support for additional sign languages.

## Author

**Developed as a Python-based Machine Learning project.**

## License

This project is available for educational and learning purposes. A formal license can be added to define reuse and distribution permissions.

# Hand Gesture Recognition System Using Machine Learning

## 📌 Overview

This project presents a machine learning-based **Hand Gesture Recognition System** using hand landmark coordinate features.

The system extracts spatial hand features and applies machine learning algorithms to recognize different hand activities and identify individuals based on unique hand characteristics.

The project contains multiple experiments evaluating gesture recognition, person identification, and joint recognition performance under different experimental conditions.

---

# 🎯 Objectives

The main objectives of this project are:

- Develop a hand gesture recognition framework using landmark-based features.
- Analyze the effectiveness of coordinate-based hand representations.
- Identify different hand activities using machine learning models.
- Perform person identification using hand landmark patterns.
- Evaluate model performance with different training ratios and dataset sizes.

---

# 🏗️ Project Structure







Hand-Gesture-Recognition/

│
├── realtime_gesture.py
├── gesture_pipeline.pkl
├── hand_landmarker.task
│
├── Experiment_1_Hand_Activity_Identification.ipynb
├── Experiment_2_Person_Identification.ipynb
├── Experiment_3_Joint_HandActivity_Person_Identification.ipynb
│
├── Experiment1_HandActivity_DataSize.csv
├── Experiment1_HandActivity_TrainingRatio.csv
│
├── Experiment2_PersonID_DataSize.csv
├── Experiment2_PersonID_TrainingRatio.csv
│
├── Experiment3_Joint_DataSize.csv
├── Experiment3_Joint_TrainingRatio.csv
│
├── Person_Identification_Results.csv
├── Experiment_Results_FINAL.xlsx
│
├── Visualization_All_Experiments.ipynb
├── Exp3_JointID_Visualization.ipynb
│
└── README.md







---

# 🧠 Methodology

## 1. Hand Landmark Extraction

The system represents a hand using landmark coordinates.

The extracted features describe:

- Finger joint positions
- Palm structure
- Spatial relationships between landmarks
- Hand geometry information

These coordinates are used as machine learning input features.

---

# 2. Machine Learning Pipeline

The extracted landmark features are processed through a machine learning pipeline:









---

# 🧪 Experiments

## Experiment 1: Hand Activity Identification

Notebook:

Purpose:

Recognize different hand gestures based on landmark coordinate features.

Evaluation performed using:

- Different training ratios
- Different dataset sizes
- Classification performance comparison


---

## Experiment 2: Person Identification

Notebook:

Purpose:

Identify individuals using unique hand landmark characteristics.

Analysis includes:

- Person classification
- Training ratio evaluation
- Dataset size analysis


---

## Experiment 3: Joint Hand Activity and Person Identification

Notebook:

Purpose:

Perform combined recognition of:

- Hand activity
- Person identity

Evaluation includes:

- Joint classification performance
- Confusion matrix analysis
- Model comparison


---

# 📊 Results and Analysis

The project includes:

- Performance comparison tables
- Dataset size analysis
- Training ratio experiments
- Classification results
- Visualization notebooks

Generated analysis files:

- CSV experiment results
- Excel result summary
- Visualization notebooks

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/basudevsaw170605-beep/Hand-Gesture-Recognition.git
cd Hand-Gesture-Recognition
python -m venv .venv
.venv\Scripts\activate


▶️ Running the Project

For real-time gesture recognition:

python realtime_gesture.py

For experiment analysis:

Open Jupyter Notebook:

jupyter notebook

Run the required experiment notebook.


🛠️ Technologies Used
Python
NumPy
Pandas
Scikit-learn
OpenCV
Jupyter Notebook
Matplotlib
🔬 Research Contribution

This project investigates the effectiveness of hand landmark coordinate features for machine learning-based recognition.

The framework explores:

Gesture recognition
Person identification
Joint recognition tasks
Human-computer interaction applications
📈 Applications

Possible applications:

Touchless computer interaction
Smart control systems
Assistive technologies
Human-computer interaction
Gesture-based interfaces
👨‍💻 Author

Basudev

Machine Learning | Computer Vision | Data Science

GitHub:

https://github.com/basudevsaw170605-beep

📄 License

This project is developed for academic and research purposes.


Then save and run:

```powershell
git add README.md
git commit -m "Add professional README"
git push

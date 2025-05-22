# # 🧠 Liver Cirrhosis Diagnosis and Food Recommender Assistant

A Machine Learning-powered medical assistant that diagnoses liver cirrhosis using MRI scans and clinical data, predicts its stage, and recommends personalized diet plans—all through a responsive Streamlit web interface.

![Liver Cirrhosis AI](https://img.shields.io/badge/Project-AIML-blue) ![Python](https://img.shields.io/badge/Python-3.8-green) ![Streamlit](https://img.shields.io/badge/Streamlit-Cloud%20UI-orange)

---

## 📌 Project Overview

Traditional liver cirrhosis diagnosis is invasive, slow, and often lacks stage-specific dietary support. This project leverages AI to:

- Detect liver cirrhosis using **3D MRI images** via a custom **CNN** model.
- Predict the **stage** (1–4) using **LightGBM** based on clinical lab values.
- Recommend stage-specific **liver-friendly food** using a **RandomForest** model.
- Provide all results instantly through an easy-to-use **Streamlit web application**.

> 🔬 Built for early-stage diagnosis, decision support, and nutritional management of liver cirrhosis patients.

---

## 🚀 Features

- 🧠 **3D CNN MRI Classifier** (94% Accuracy)
- 📊 **LightGBM Stage Predictor** using structured clinical data
- 🍎 **Diet Recommender** based on disease stage (Rule-Based + ML)
- 🌐 **Streamlit Web App** for real-time diagnosis & suggestions
- 📥 Upload MRI + clinical data, and get instant results
- 📈 Model visualizations and performance reports

---

## 🛠️ Tech Stack

| Technology     | Description                                      |
|----------------|--------------------------------------------------|
| `Python 3.8`   | Core programming language                        |
| `PyTorch`      | Deep Learning framework (MRI CNN)               |
| `LightGBM`     | Gradient boosting model for stage prediction     |
| `RandomForest` | Food recommendation classifier                   |
| `NiBabel`      | Load `.nii` MRI medical imaging files            |
| `scikit-learn` | Model metrics, evaluation tools                  |
| `Streamlit`    | Web interface for input and output               |
| `Pandas`       | Data handling                                    |
| `Matplotlib`   | Visualizations                                   |
| `Joblib`       | Model serialization                              |

---

## 📂 Folder Structure

```
Liver-Cirrhosis-AI/
│
├── Liver_final.ipynb               # MRI CNN model training
├── clinical_stage_model.ipynb      # Stage prediction (LightGBM)
├── food_recommender_model.ipynb    # Food recommender (Random Forest)
├── app.py                          # Streamlit web application
├── models/                         # Saved .pkl or .pth models
├── data/                           # Sample datasets
└── README.md                       # Project description
```

---

## 📊 Dataset Links

> Note: You must be logged into Google Drive or Kaggle to access datasets.


- 🧪 **MRI Scans Dataset (.nii format)**:  
  [📁 Google Drive Link]((https://drive.google.com/drive/folders/12059rgR_v7K9n_xK1QKqYyLhJkITy1FU?usp=drive_link))

- 📋 **Clinical Data and Dietery Datasets (CSV)**:  
  [📁 Google Drive Link]((https://drive.google.com/drive/folders/18YxKo7OCuYefpZPa7R3o4i9QhYZxXX-o?usp=drive_link))

- **Trained Models**:
  [📁 Google Drive Link]((https://drive.google.com/drive/folders/18YxKo7OCuYefpZPa7R3o4i9QhYZxXX-o?usp=drive_link))

  
---

## 💻 Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Liver-Cirrhosis-AI.git
cd Liver-Cirrhosis-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit App
streamlit run app.py
```

> 💡 Make sure your system has Python 3.8+ installed and MRI `.nii` files are placed in the correct directory structure.

---



## 📈 Model Performance

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| CNN (MRI)      | 94.2%    | 0.93      | 0.94   | 0.94     |
| LightGBM       | 91.3%    | 0.91      | 0.90   | 0.90     |
| RandomForest   | 88.5%    | 0.89      | 0.87   | 0.88     |

---

## 📌 Future Enhancements

- ✅ Deploy app on **Streamlit Cloud / Render**
- 📱 Mobile responsive version
- 🧬 Integration with blood test image reports
- 📡 Real-time REST API for hospital use

---

## 👨‍💻 Developed By

**Your Name**  
B.Tech Artificial Intelligence and Machine Learning  
[📧 your.email@example.com](mailto:lallithkavi@gmail.com)  
[🔗 LinkedIn](https://linkedin.com/in/lallith-ar-cr7) • [🐙 GitHub](https://github.com/DangerCR7)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

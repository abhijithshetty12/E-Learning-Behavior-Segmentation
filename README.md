# 🎓 E-Learning Student Behavioral Segmentation Engine

[![Live Demo](https://img.shields.io/badge/Live_Demo-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://e-learning-behavior-segmentation.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat&logo=jupyter&logoColor=white)

**E-Learning Student Behavioral Segmentation Engine** is an unsupervised machine learning system designed to identify distinct student learning patterns from e-learning interaction logs.

Built using **Python, K-Modes Clustering, Pandas, Scikit-learn, and Streamlit**, the project transforms raw student activity data from the **Open University Learning Analytics Dataset (OULAD)** into categorical behavioral profiles and groups learners into meaningful segments based on their engagement, consistency, assessment activity, submission behavior, and inactivity patterns.

The system also provides an interactive dashboard for exploring behavioral segments, comparing academic outcomes, and assigning new student profiles to the most relevant behavioral cluster.

---

## 🚀 Key Features

* **🎯 Behavioral Segmentation**: Groups students into meaningful learning-behavior segments using K-Modes clustering.
* **📚 Real Learning Analytics Dataset**: Uses the Open University Learning Analytics Dataset (OULAD) containing millions of student interaction records.
* **🧠 Categorical K-Modes Clustering**: Uses K-Modes instead of K-Means because the final behavioral features are categorical.
* **⚙️ Behavioral Feature Engineering**: Converts raw interaction logs into meaningful categorical learning indicators.
* **📊 Cluster Evaluation**: Compares multiple values of `K` using K-Modes cost, elbow analysis, Hamming-distance silhouette score, cluster balance, and behavioral interpretability.
* **🔎 Segment Explorer**: View the typical behavioral characteristics of each discovered student group.
* **📈 Academic Outcome Analysis**: Compares behavioral segments with Pass, Fail, Distinction, and Withdrawn outcomes without using those outcomes during clustering.
* **🧑‍🎓 Student Segment Prediction**: Enter a new student's behavioral profile and assign it to one of the learned behavioral segments using the saved K-Modes model.
* **💻 Interactive Dashboard**: Streamlit-based interface for visualizing and demonstrating the complete ML system.

---

## 🧠 Behavioral Features

The final clustering model uses **8 categorical behavioral features**:

| Feature | Categories |
| :--- | :--- |
| **Activity Level** | Low, Medium, High |
| **Learning Frequency** | Rare, Moderate, Frequent |
| **Learning Consistency** | Irregular, Moderately Regular, Consistent |
| **Content Preference** | Learning Content, Discussion, Assessment, Navigation |
| **Resource Diversity** | Narrow, Moderate, Diverse |
| **Assessment Engagement** | Low, Medium, High |
| **Submission Behaviour** | Early, On-Time, Late, No Submission Data |
| **Inactivity Pattern** | Low, Moderate, High Inactivity |

---

## 👥 Discovered Behavioral Segments

The final K-Modes model uses **4 behavioral clusters**:

### 🟢 Highly Engaged Consistent Learners
Students with frequent platform activity, consistent learning patterns, diverse resource usage, high assessment engagement, and low inactivity.

### 🔵 Steady Assessment-Engaged Learners
Students with moderate and stable learning activity, strong assessment participation, and generally timely submissions.

### 🟠 Irregular Late-Submission Learners
Students with inconsistent learning activity, higher inactivity, and a tendency toward late submissions.

### 🔴 Low-Engagement Inactive Learners
Students with low activity, rare platform usage, limited resource diversity, low assessment engagement, and high inactivity.

---

## 📊 Dataset

The project uses the **Open University Learning Analytics Dataset (OULAD)**.

The dataset contains student demographic information, assessment records, course information, registration data, learning-resource metadata, and detailed Virtual Learning Environment interaction logs.

### Main Files Used

| Dataset | Purpose |
| :--- | :--- |
| `studentVle.csv` | Student interaction and click activity |
| `vle.csv` | Learning resource and activity types |
| `studentAssessment.csv` | Student assessment submissions |
| `assessments.csv` | Assessment metadata and deadlines |
| `studentInfo.csv` | Student and academic outcome information |
| `studentRegistration.csv` | Registration and withdrawal information |
| `courses.csv` | Course presentation duration |

After preprocessing and aggregation, the system produces approximately **29,228 student-course behavioral profiles**.

---

## 🔄 ML Workflow

```text
OULAD Raw Dataset
        ↓
Data Exploration
        ↓
Data Cleaning & Aggregation
        ↓
Behavioral Feature Engineering
        ↓
Categorical Student Profiles
        ↓
K-Modes Clustering
        ↓
K Selection & Evaluation
        ↓
Behavioral Segment Interpretation
        ↓
Academic Outcome Analysis
        ↓
Saved ML Model
        ↓
Streamlit Dashboard
```

---

## 📐 Model Selection

K-Modes clustering was evaluated for cluster counts from **K = 2 to K = 8**.

| K | Cost | Silhouette Score |
| :---: | ---: | ---: |
| 2 | 92,168 | 0.3462 |
| 3 | 72,718 | **0.3556** |
| 4 | 66,527 | 0.3117 |
| 5 | 64,530 | 0.2751 |
| 6 | 60,005 | 0.2319 |
| 7 | 57,686 | 0.2173 |
| 8 | 56,547 | 0.1944 |

Although **K = 3 achieved the highest silhouette score**, **K = 4** was selected for the final model because it produced a more meaningful and interpretable separation of student behavioral patterns while maintaining well-balanced cluster sizes.

Final cluster distribution:

| Behavioral Segment | Approx. Share |
| :--- | ---: |
| Low-Engagement Inactive Learners | 31.26% |
| Highly Engaged Consistent Learners | 29.50% |
| Steady Assessment-Engaged Learners | 20.71% |
| Irregular Late-Submission Learners | 18.52% |

---

## 📈 Academic Outcome Analysis

Academic outcomes were **not used as clustering features**. They were analyzed only after clustering to evaluate whether the discovered behavioral segments corresponded with meaningful differences in student outcomes.

| Behavioral Segment | Distinction | Pass | Fail | Withdrawn |
| :--- | ---: | ---: | ---: | ---: |
| **Highly Engaged Consistent Learners** | 24.53% | 65.91% | 5.90% | 3.66% |
| **Steady Assessment-Engaged Learners** | 10.18% | 67.88% | 11.28% | 10.66% |
| **Irregular Late-Submission Learners** | 4.34% | 34.61% | 33.62% | 27.43% |
| **Low-Engagement Inactive Learners** | 0.63% | 7.57% | 40.12% | 51.67% |

These results show a strong association between learning behavior and academic outcomes. However, the analysis represents **correlation rather than causation**.

---

## 🛠 Tech Stack

| Category | Tools |
| :--- | :--- |
| **Programming Language** | Python |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | K-Modes |
| **Evaluation** | Scikit-learn |
| **Visualization** | Plotly, Matplotlib, Streamlit |
| **Development** | Jupyter Notebook, VS Code |
| **Model Persistence** | Joblib |
| **Dataset** | OULAD |

---

## 📁 Project Structure

```text
E-Learning-Behavior-Segmentation/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── assessments.csv
│   │   ├── courses.csv
│   │   ├── studentAssessment.csv
│   │   ├── studentInfo.csv
│   │   ├── studentRegistration.csv
│   │   ├── studentVle.csv
│   │   └── vle.csv
│   │
│   └── processed/
│       ├── student_behaviour.csv
│       └── student_behaviour_clustered.csv
│
├── models/
│   ├── kmodes_model.pkl
│   └── model_metadata.pkl
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_kmodes_clustering.ipynb
│
├── screenshots/
│   ├── dashboard.png
│   ├── overview.png
│   ├── segments.png
│   ├── outcomes.png
│   ├── prediction.png
│   └── about.png
│
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   └── clustering.py
│
├── requirements.txt
└── README.md
```

---

## 🖼️ Interface Preview

### Main Dashboard

<p align="center">
  <img src="/screenshots/dashboard.png" alt="Main Dashboard" width="100%" />
</p>

### Dashboard Sections

| **Overview** | **Behavioral Segments** |
| :---: | :---: |
| <img src="/screenshots/overview.png" alt="Overview" /> | <img src="/screenshots/segments.png" alt="Behavioral Segments" /> |

| **Academic Outcomes** | **Student Segment Prediction** |
| :---: | :---: |
| <img src="/screenshots/outcomes.png" alt="Academic Outcomes" /> | <img src="/screenshots/prediction.png" alt="Student Segment Prediction" /> |

### About & Methodology

<p align="center">
  <img src="/screenshots/about.png" alt="About and Methodology" width="100%" />
</p>

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/abhijithshetty12/E-Learning-Behavior-Segmentation.git
```

Navigate into the project:

```bash
cd E-Learning-Behavior-Segmentation
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

From the project root:

```bash
streamlit run app/app.py
```

The Streamlit dashboard will open in your browser at:

```text
http://localhost:8501
```

---

## 🧪 ML Notebooks

The project is divided into three main notebooks:

### `01_data_exploration.ipynb`

Explores the OULAD dataset, its structure, activity types, assessments, interaction logs, and relationships between datasets.

### `02_feature_engineering.ipynb`

Transforms millions of raw VLE interactions into categorical behavioral profiles suitable for K-Modes clustering.

### `03_kmodes_clustering.ipynb`

Performs K-Modes clustering, evaluates different values of K, interprets the resulting student segments, analyzes academic outcomes, and saves the final model.

---

## 🎯 Project Objective

The main objective of this project is to demonstrate how unsupervised machine learning can be used to identify distinct student learning behaviors from e-learning interaction data.

Rather than predicting marks or directly classifying students as Pass or Fail, the system focuses on discovering **behavioral patterns** that can help educators better understand different types of learners and identify groups that may require additional academic support.

---

## ⚠️ Limitations

* The final behavioral signals are discretized into categorical levels, which simplifies some continuous interaction patterns.
* The learned segments are derived from **OULAD** and may not transfer directly to every institution, LMS, course structure, or learner population.
* Academic outcomes are used only for post-clustering analysis; the observed relationships represent **association, not causation**.
* Live prediction expects the **8 engineered categorical behavioral features** rather than raw clickstream, timestamp, or assessment records.
* The current application is an offline analytical prototype and is not connected to a real-time Learning Management System.

---

## 🔮 Future Scope

* Real-time integration with Learning Management Systems.
* Automatic generation of student behavioral profiles from live activity logs.
* Personalized learning recommendations for each behavioral segment.
* Early-warning system for potentially disengaged learners.
* Instructor dashboard with course-level behavioral analytics.
* Comparison with other clustering algorithms such as K-Prototypes and hierarchical clustering.
* Longitudinal analysis of student behavior across multiple courses and semesters.

---

## 👨‍💻 Author

**Abhijith Shetty**  
*AI & Machine Learning Student | Developer*

> "Building intelligent systems that transform data into meaningful and practical insights."

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/abhijithshetty12)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/abhijithshetty12)

---

## 🌟 Show Your Support

If you find this project interesting or useful, consider giving it a ⭐ on **GitHub**.

It helps support the project and encourages further improvements.

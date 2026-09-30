from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="E-Learning Behavioral Segmentation",
    page_icon="🎓",
    layout="wide"
)


# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "student_behaviour_clustered.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "kmodes_model.pkl"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.pkl"
)


# ---------------------------------------------------------
# Load Data and Model
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_metadata():
    return joblib.load(METADATA_PATH)


try:
    df = load_data()
    model = load_model()
    metadata = load_metadata()

except FileNotFoundError as error:
    st.error(
        "Required project files were not found. "
        "Make sure the trained model, metadata, and clustered dataset exist."
    )

    st.code(str(error))

    st.stop()


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

BEHAVIOUR_FEATURES = metadata["features"]
CLUSTER_NAMES = metadata["cluster_names"]
N_CLUSTERS = metadata["n_clusters"]


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title(
    "🎓 E-Learning Student Behavioral Segmentation Engine"
)

st.write(
    """
    An unsupervised learning system that analyzes student
    interaction patterns and groups learners into behavioral
    segments using **K-Modes clustering**.
    """
)


# ---------------------------------------------------------
# Overview Metrics
# ---------------------------------------------------------

st.subheader("Overview")

total_profiles = len(df)

unique_students = df["id_student"].nunique()

modules = df["code_module"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Behavioral Profiles",
    f"{total_profiles:,}"
)

col2.metric(
    "Unique Students",
    f"{unique_students:,}"
)

col3.metric(
    "Behavioral Segments",
    N_CLUSTERS
)

col4.metric(
    "Modules",
    modules
)


# ---------------------------------------------------------
# Cluster Distribution
# ---------------------------------------------------------

st.divider()

st.subheader("Behavioral Segment Distribution")

cluster_distribution = (
    df["Cluster_Name"]
    .value_counts()
    .rename_axis("Behavioral Segment")
    .reset_index(name="Students")
)

cluster_distribution["Percentage"] = (
    cluster_distribution["Students"]
    / cluster_distribution["Students"].sum()
    * 100
).round(2)


left, right = st.columns([2, 1])

with left:

    chart_data = (
        cluster_distribution
        .set_index("Behavioral Segment")["Students"]
    )

    st.bar_chart(chart_data)


with right:

    st.dataframe(
        cluster_distribution,
        hide_index=True,
        use_container_width=True
    )


# ---------------------------------------------------------
# Segment Explorer
# ---------------------------------------------------------

st.divider()

st.subheader("🔎 Behavioral Segment Explorer")

segment_names = sorted(
    df["Cluster_Name"].dropna().unique()
)

selected_segment = st.selectbox(
    "Select a behavioral segment",
    segment_names
)

segment_data = df[
    df["Cluster_Name"] == selected_segment
]

cluster_id = int(
    segment_data["Cluster"].iloc[0]
)

st.write(
    f"**Cluster ID:** {cluster_id}"
)

st.write(
    f"**Profiles in segment:** {len(segment_data):,}"
)


# ---------------------------------------------------------
# Display Cluster Mode
# ---------------------------------------------------------

cluster_mode = pd.DataFrame(
    model.cluster_centroids_,
    columns=BEHAVIOUR_FEATURES
)

cluster_mode.index.name = "Cluster"

selected_mode = (
    cluster_mode
    .loc[cluster_id]
)

mode_table = pd.DataFrame({
    "Behavioral Feature": selected_mode.index,
    "Typical Behavior": selected_mode.values
})

st.markdown("### Typical Behavioral Profile")

st.dataframe(
    mode_table,
    hide_index=True,
    use_container_width=True
)


# ---------------------------------------------------------
# Cluster Interpretation
# ---------------------------------------------------------

SEGMENT_DESCRIPTIONS = {

    "Highly Engaged Consistent Learners":
        """
        Students with frequent platform activity, consistent
        learning patterns, diverse resource usage and strong
        assessment engagement.
        """,

    "Steady Assessment-Engaged Learners":
        """
        Students showing moderate and relatively stable platform
        participation while maintaining strong engagement with
        assessments and generally timely submissions.
        """,

    "Irregular Late-Submission Learners":
        """
        Students displaying inconsistent learning activity,
        noticeable inactivity periods and a tendency toward
        late submissions.
        """,

    "Low-Engagement Inactive Learners":
        """
        Students with low activity, infrequent platform usage,
        limited resource diversity and substantial inactivity.
        """
}

description = SEGMENT_DESCRIPTIONS.get(
    selected_segment,
    "Behavioral segment identified through K-Modes clustering."
)

st.info(description)


# ---------------------------------------------------------
# Academic Outcome Analysis
# ---------------------------------------------------------

st.divider()

st.subheader("📊 Academic Outcomes by Behavioral Segment")

outcome_percentage = pd.crosstab(
    df["Cluster_Name"],
    df["final_result"],
    normalize="index"
).mul(100).round(2)

st.bar_chart(outcome_percentage)

st.dataframe(
    outcome_percentage,
    use_container_width=True
)

st.caption(
    """
    Academic outcomes were not used to create the clusters.
    They are shown here only for post-clustering analysis.
    Differences therefore represent associations rather than
    causal relationships.
    """
)


# ---------------------------------------------------------
# Selected Segment Outcome Details
# ---------------------------------------------------------

st.markdown("### Selected Segment Outcomes")

selected_outcomes = (
    segment_data["final_result"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .rename_axis("Final Result")
    .reset_index(name="Percentage")
)

st.dataframe(
    selected_outcomes,
    hide_index=True,
    use_container_width=True
)


# ---------------------------------------------------------
# Prediction Section
# ---------------------------------------------------------

st.divider()

st.subheader("🧠 Predict Student Behavioral Segment")

st.write(
    """
    Enter a student's behavioral profile below.
    The trained K-Modes model will assign the profile to one
    of the discovered behavioral segments.
    """
)


# ---------------------------------------------------------
# Possible Categories
# ---------------------------------------------------------

feature_options = {

    "Activity_Level": [
        "Low",
        "Medium",
        "High"
    ],

    "Learning_Frequency": [
        "Rare",
        "Moderate",
        "Frequent"
    ],

    "Learning_Consistency": [
        "Irregular",
        "Moderately Regular",
        "Consistent"
    ],

    "Content_Preference": [
        "Learning Content",
        "Discussion",
        "Assessment",
        "Navigation"
    ],

    "Resource_Diversity": [
        "Narrow",
        "Moderate",
        "Diverse"
    ],

    "Assessment_Engagement": [
        "Low",
        "Medium",
        "High"
    ],

    "Submission_Behaviour": [
        "Early",
        "On-Time",
        "Late",
        "No Submission Data"
    ],

    "Inactivity_Pattern": [
        "Low Inactivity",
        "Moderate Inactivity",
        "High Inactivity"
    ]
}


# ---------------------------------------------------------
# Prediction Form
# ---------------------------------------------------------

with st.form("student_prediction_form"):

    col1, col2 = st.columns(2)

    user_input = {}

    with col1:

        user_input["Activity_Level"] = st.selectbox(
            "Activity Level",
            feature_options["Activity_Level"]
        )

        user_input["Learning_Frequency"] = st.selectbox(
            "Learning Frequency",
            feature_options["Learning_Frequency"]
        )

        user_input["Learning_Consistency"] = st.selectbox(
            "Learning Consistency",
            feature_options["Learning_Consistency"]
        )

        user_input["Content_Preference"] = st.selectbox(
            "Content Preference",
            feature_options["Content_Preference"]
        )

    with col2:

        user_input["Resource_Diversity"] = st.selectbox(
            "Resource Diversity",
            feature_options["Resource_Diversity"]
        )

        user_input["Assessment_Engagement"] = st.selectbox(
            "Assessment Engagement",
            feature_options["Assessment_Engagement"]
        )

        user_input["Submission_Behaviour"] = st.selectbox(
            "Submission Behaviour",
            feature_options["Submission_Behaviour"]
        )

        user_input["Inactivity_Pattern"] = st.selectbox(
            "Inactivity Pattern",
            feature_options["Inactivity_Pattern"]
        )

    submitted = st.form_submit_button(
        "Predict Behavioral Segment",
        use_container_width=True
    )


# ---------------------------------------------------------
# Make Prediction
# ---------------------------------------------------------

if submitted:

    input_df = pd.DataFrame([
        [
            user_input[feature]
            for feature in BEHAVIOUR_FEATURES
        ]
    ],
        columns=BEHAVIOUR_FEATURES
    )

    predicted_cluster = int(
        model.predict(input_df)[0]
    )

    predicted_segment = CLUSTER_NAMES[
        predicted_cluster
    ]

    st.success(
        f"Predicted Segment: {predicted_segment}"
    )

    st.write(
        f"**Cluster ID:** {predicted_cluster}"
    )

    st.markdown(
        "### Entered Behavioral Profile"
    )

    prediction_profile = pd.DataFrame({
        "Feature": BEHAVIOUR_FEATURES,
        "Selected Value": [
            user_input[feature]
            for feature in BEHAVIOUR_FEATURES
        ]
    })

    st.dataframe(
        prediction_profile,
        hide_index=True,
        use_container_width=True
    )

    prediction_description = (
        SEGMENT_DESCRIPTIONS.get(
            predicted_segment
        )
    )

    if prediction_description:
        st.info(prediction_description)


# ---------------------------------------------------------
# Methodology
# ---------------------------------------------------------

st.divider()

with st.expander("ℹ️ About the Model"):

    st.markdown(
        """
        ### Methodology

        **Dataset:** Open University Learning Analytics Dataset
        (OULAD)

        **Algorithm:** K-Modes Clustering

        **Number of clusters:** 4

        **Behavioral features:**

        - Activity Level
        - Learning Frequency
        - Learning Consistency
        - Content Preference
        - Resource Diversity
        - Assessment Engagement
        - Submission Behaviour
        - Inactivity Pattern

        K-Modes was selected because the final behavioral
        features are categorical rather than continuous.

        The number of clusters was evaluated using K-Modes cost,
        silhouette analysis with Hamming distance, cluster size,
        and behavioral interpretability.
        """
    )
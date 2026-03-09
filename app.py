import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="World Development Clustering", layout="centered")

# ---------------- LOAD MODELS ---------------- #
@st.cache_resource
def load_models():
    scaler = pickle.load(open("scaler.pkl", "rb"))
    kmeans = pickle.load(open("kmeans.pkl", "rb"))
    gmm = pickle.load(open("gmm.pkl", "rb"))
    birch = pickle.load(open("birch.pkl", "rb"))
    feature_names = pickle.load(open("feature_names.pkl", "rb"))
    return scaler, kmeans, gmm, birch, feature_names

scaler, kmeans, gmm, birch, feature_names = load_models()

# ---------------- TITLE ---------------- #
st.title("Clustering App")
st.markdown("Assign development clusters based on socio-economic indicators.")

# ---------------- MODEL SELECTION ---------------- #
model_choice = st.selectbox(
    "Choose Clustering Model",
    ["K-Means", "Gaussian Mixture Model", "BIRCH"]
)

# ---------------- INPUT SECTION ---------------- #
st.subheader("Enter Country Indicators")

col1, col2 = st.columns(2)

with col1:
    gdp = st.number_input(
        "GDP",
        min_value=0.0,
        value=10000.0,
        step=1000.0
    )

    life_male = st.slider(
        "Life Expectancy Male",
        min_value=0,
        max_value=100,
        value=70
    )

    infant = st.slider(
        "Infant Mortality Rate",
        min_value=0,
        max_value=150,
        value=20
    )

with col2:
    life_female = st.slider(
        "Life Expectancy Female",
        min_value=0,
        max_value=100,
        value=75
    )

    internet = st.slider(
        "Internet Usage (%)",
        min_value=0,
        max_value=100,
        value=60
    )

    population = st.number_input(
        "Population Total",
        min_value=0.0,
        value=10000000.0,
        step=1000000.0
    )

# ---------------- BUILD INPUT DATAFRAME ---------------- #
user_input = {
    "GDP": gdp,
    "Life Expectancy Male": life_male,
    "Life Expectancy Female": life_female,
    "Infant Mortality Rate": infant,
    "Internet Usage": internet,
    "Population Total": population
}

input_df = pd.DataFrame([user_input])
input_df = input_df.reindex(columns=feature_names)

# ---------------- CLUSTER INTERPRETATION ---------------- #
def interpret_cluster(cluster):
    if cluster == 0:
        return "🟢 Developed Country"
    elif cluster == 1:
        return "🟡 Developing Country"
    elif cluster == 2:
        return "🔴 Underdeveloped Country"
    else:
        return "Unknown"

# ---------------- PREDICTION ---------------- #
if st.button("Assign Cluster"):

    try:
        input_scaled = scaler.transform(input_df)

        if model_choice == "K-Means":
            cluster = kmeans.predict(input_scaled)[0]
        elif model_choice == "Gaussian Mixture Model":
            cluster = gmm.predict(input_scaled)[0]
        elif model_choice == "BIRCH":
            cluster = birch.predict(input_scaled)[0]

        label = interpret_cluster(cluster)

        st.success(f"📊 Assigned Cluster: {cluster}")
        st.info(f"Country Category: {label}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.subheader("Models Used")

st.markdown("""
✔ **K-Means** (Centroid-based clustering)  
✔ **Gaussian Mixture Model** (Probabilistic clustering)  
✔ **BIRCH** (Hierarchical-based scalable clustering)  
✔ Hierarchical Clustering (Analysis only)  
✔ DBSCAN (Analysis only – no predict method)  
""")

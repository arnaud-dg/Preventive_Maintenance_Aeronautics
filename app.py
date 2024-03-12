import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import pickle
# import shap
import requests
import joblib
import streamlit.components.v1 as components

st.set_page_config(layout="wide", 
                   page_icon="🛠️", 
                   page_title="Pred. Maintenance")
# st.title('🛠️ Predictive Maintenance Interface')
st.markdown("<h1 style='text-align: center; color: #323232;'>🛠️ Predictive Maintenance Interface 📉</h1>", unsafe_allow_html=True)
# shap.initjs()

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    features = ['setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11',
            's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','label_mcc']
    df = df[features]
    columns_title = ["setting_1", "setting_2", "setting_3", "(Fan inlet temperature) (◦R)", "(LPC outlet temperature) (◦R)",
        "(HPC outlet temperature) (◦R)", "(LPT outlet temperature) (◦R)", "(Fan inlet Pressure) (psia)", "(bypass-duct pressure) (psia)",
        "(HPC outlet pressure) (psia)", "(Physical fan speed) (rpm)", "(Physical core speed) (rpm)", "(Engine pressure ratio(P50/P2)", 
        "(HPC outlet Static pressure) (psia)", "(Ratio of fuel flow to Ps30) (pps/psia)", "(Corrected fan speed) (rpm)", "(Corrected core speed) (rpm)",
        "(Bypass Ratio) ", "(Burner fuel-air ratio)", "(Bleed Enthalpy)", "(Required fan speed)", "(Required fan conversion speed)", "(High-pressure turbines Cool air flow)",
        "(Low-pressure turbines Cool air flow)", "Label"]
    df.columns = columns_title

    # Calculate simple stats about the dataframe
    stats_df = pd.DataFrame({
        'Mean': df.mean(),
        'Std_dev': df.std(),
        'Min': df.min(),
        'Max': df.max()
    })
    return df, stats_df

# Create a set of values (sampling + randomization)
def sample_selection(df, stats_df):
    # Sampling
    sample = pd.concat([df[df['Label'] == 0].sample(20), df[df['Label'] == 1].sample(20), df[df['Label'] == 2].sample(20)], axis=0)
    sample = sample.sample(1)
    sample = sample.drop('Label', axis=1)
    
    # Randomization of new values close to the sampled ones
    for i, col in enumerate(sample.columns.to_list()):
        std = stats_df.loc[col, 'Std_dev'] / 10
        sample[col] = sample[col] + np.random.normal(0, std)

    return sample

df, stats_df = load_data()

# Intro
st.sidebar.header("Introduction")
st.sidebar.write("This web application aims to illustrate the **Predictive Maintenance project** proposed by [Arnaud Duigou](https://arnaud-dg.github.io/). This project involves to predict the risk of a machine breakdown using Machine Learning tools.")
st.sidebar.divider()
st.sidebar.header("How does it works")
st.sidebar.write("- Data from 24 sensors connected to turbines were used to train a Machine Learning model. This training data, after a dimension reduction step, results in the 3D scatterplot shown here. Each record is associated with a risk of breakdown corresponding to the colors: **:red[Red (High Risk)]**, **:orange[Amber (Low Risk)]**, **:green[Green (Standard state)]**.")
st.sidebar.write("- When you click on the **Simulate new data** button, the program will randomly generate a new configuration of values that does not exist in the training dataset.")
st.sidebar.write("- This will simulate a new data transmission from a machine, and this new state will appear in **:violet[purple]** on the SD scatterplot. The Machine Learning classification model will then make a prediction to indicate the associated risk level, as well as the maintenance actions to be taken.")
st.sidebar.divider()
st.sidebar.header("Detailed article")
st.sidebar.write("If you want to learn furthermore about this project, please consult the medium article : weblink to replace")

# Display the results
sample = sample_selection(df, stats_df)
st.header("Inputs")
#col1, col2 = st.columns([1, 6])
st.button('**Simulate new data**')
# with col1:
#     st.button('Simulate new data')
# with col2:
#     st.markdown("The new configuration of values will be visible in **:violet[violet]** on the PCA chart, and a prediction will be made to determine the associated risk of failure.", unsafe_allow_html=True)
with st.expander("New Simulated dataset/configuration", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        df1 = sample.reset_index().iloc[0, 0:6]
        st.dataframe(df1, use_container_width=True, column_config={"0": "Values"})
    with col2:
        df2 = sample.reset_index().iloc[0, 6:12]
        # df2 = df2.rename(index={'0':'Values'}, inplace=True)
        st.dataframe(df2, use_container_width=True, column_config={"0": "Values"})
    with col3:
        df3 = sample.reset_index().iloc[0, 12:18]
        # df3 = df3.rename(index={'0':'Values'}, inplace=True)
        st.dataframe(df3, use_container_width=True, column_config={"0": "Values"})
    with col4:
        df4 = sample.reset_index().iloc[0, 18:24]
        # df4 = df4.rename(index={'0':'Values'}, inplace=True)
        st.dataframe(df4, use_container_width=True, column_config={"0": "Values"})
st.divider()

# Graphical Results
st.header("Modeling results")
col1, col2 = st.columns([2, 1])
with col1: # PCA
   st.subheader("Data Visualization - After Dimensionality Reduction")
   # Normlization of the features
   columns_title = ["setting_1", "setting_2", "setting_3", "(Fan inlet temperature) (◦R)", "(LPC outlet temperature) (◦R)",
        "(HPC outlet temperature) (◦R)", "(LPT outlet temperature) (◦R)", "(Fan inlet Pressure) (psia)", "(bypass-duct pressure) (psia)",
        "(HPC outlet pressure) (psia)", "(Physical fan speed) (rpm)", "(Physical core speed) (rpm)", "(Engine pressure ratio(P50/P2)", 
        "(HPC outlet Static pressure) (psia)", "(Ratio of fuel flow to Ps30) (pps/psia)", "(Corrected fan speed) (rpm)", "(Corrected core speed) (rpm)",
        "(Bypass Ratio) ", "(Burner fuel-air ratio)", "(Bleed Enthalpy)", "(Required fan speed)", "(Required fan conversion speed)", "(High-pressure turbines Cool air flow)",
        "(Low-pressure turbines Cool air flow)"]
   target = df['Label']
   X = pd.concat([df[columns_title], sample], axis=0)
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   # Instanciation PCA with 3 components
   pca = PCA(n_components=3)
   pca.fit(X_scaled)
   X_proj = pca.transform(X_scaled)
   target = df['Label']
   target = [str(i) for i in target]
   target.append('3')

   df_PCA = pd.DataFrame({
    'Component PC1': X_proj[:, 0],
    'Component PC2': X_proj[:, 1],
    'Component PC3': X_proj[:, 2],
    'Target': target
    })
   df_PCA['Size'] = df_PCA['Target'].apply(lambda x: 10 if x == '3' else 2)
   color_discrete_map = {'0': 'rgba(0,112,0, 0.05)', '1': 'rgba(255,191,0, 0.2)', 
                        '2': 'rgba(210,34,45, 0.2)', '3': 'rgba(68,4,116, 1)'}
   # Création du graphique 3D
   fig = px.scatter_3d(df_PCA, x='Component PC1', y='Component PC2', z='Component PC3',
                        color='Target', size='Size',
                        color_discrete_map=color_discrete_map)
   # Mise à jour des légendes et titres
   fig.update_layout(legend_title="Classes",
                    scene=dict(
                        xaxis_title='Component PC1',
                        yaxis_title='Component PC2',
                        zaxis_title='Component PC3', 
                    ))
   camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=-0.3),
    eye=dict(x=0, y=-1.75, z=0.75))
   fig.update_layout(scene_camera=camera)
   fig.update_traces(marker=dict(line_width=0))
   fig.update_traces(selector=dict(name='0'), name='Normal', marker=dict(color='green'))
   fig.update_traces(selector=dict(name='1'), name='Low risk', marker=dict(color='orange'))
   fig.update_traces(selector=dict(name='2'), name='High risk', marker=dict(color='red'))
   fig.update_traces(selector=dict(name='3'), showlegend=False)
   fig.update_layout(title={
        'text': "3D Scatterplot - Principal Component Analysis",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
   fig.update(layout_coloraxis_showscale=False)
   fig.update_layout(width=800, height=600)
   st.plotly_chart(fig)

with col2: # Prediction + SHAP
   st.subheader("Model Predictions")
   url_modele = "https://github.com/arnaud-dg/Preventive_Maintenance_Aeronautics/raw/main/best_model.pkl"
   reponse = requests.get(url_modele)
   open("best_model.pkl", "wb").write(reponse.content)
   loaded_model = joblib.load("best_model.pkl")
   sample_to_predict = X_scaled[-1,:]
   prediction=loaded_model.predict_proba(sample_to_predict.reshape(1, -1))
   normal = prediction[0][0]
   low = prediction[0][1]
   high = prediction[0][2]
   if max([normal, low, high]) == normal:
       config = 1
   elif max([normal, low, high]) == low:
       config = 2
   else:
       config = 3
       prediction_text = "High risk"

   st.slider('Normal condition proba.', 0.0, 1.0, normal)
   st.slider('Low risk of failure proba.', 0.0, 1.0, low)
   st.slider('High risk of failure proba.', 0.0, 1.0, high)
   if config == 1:
    #    st.markdown("The new point is predicted as **:green[Normal condition]**.")
       st.success("The current state is predicted as **Normal condition**. Maintenance is **not required**.")
   elif config == 2:
       st.warning("The current state is predicted as **Low risk of failure**. It is ""required to plan** an intervention.")
    #    st.markdown("The new point is predicted with a **:orange[Low risk of failure]**.")
   else:
       st.error("The current state is predicted as **High risk of failure**. Maintenance is **required**, as soon as possible.")
    #    st.markdown("The new point is predicted with a **:red[High risk of failure]**.")

st.divider()

# def st_shap(plot, height=None):
#    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
#    components.html(shap_html, height=height)
# XAI
# st.header("Explainability of the model")
# compute SHAP values
# explainer = shap.TreeExplainer(loaded_model)
# choosen_instance = sample_to_predict.reshape(1, -1)
# shap_values = explainer.shap_values(choosen_instance)
# st.pyplot(shap.plots.text(shap_values))
# st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance, matplotlib=True))

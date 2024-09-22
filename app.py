import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from feature_engineering import FeatureEngineer
import pickle




# Set up a sidebar menu to navigate between pages
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Predict Forest Cover", "Data Analytics", "Model Evaluation"])

# Load the trained model
model = pickle.load(open('forest_cover_model.pkl', 'rb'))

# Initialize the FeatureEngineer class
fe = FeatureEngineer()

# Function to collect user inputs for prediction
def user_input_features():
    st.sidebar.header('User Input Features')
    elevation = st.sidebar.slider('Elevation', 0, 5000, 2500)
    aspect = st.sidebar.slider('Aspect', 0, 360, 180)
    slope = st.sidebar.slider('Slope', 0, 90, 10)
    horizontal_distance_hydrology = st.sidebar.slider('Horizontal Distance to Hydrology', 0, 500, 250)
    vertical_distance_hydrology = st.sidebar.slider('Vertical Distance to Hydrology', 0, 500, 50)
    horizontal_distance_roadways = st.sidebar.slider('Horizontal Distance to Roadways', 0, 10000, 500)
    horizontal_distance_fire_points = st.sidebar.slider('Horizontal Distance to Fire Points', 0, 10000, 4113)
    hillshade_9am = st.sidebar.slider('Hillshade at 9 AM', 0, 255, 200)
    hillshade_noon = st.sidebar.slider('Hillshade at Noon', 0, 255, 220)
    hillshade_3pm = st.sidebar.slider('Hillshade at 3 PM', 0, 255, 150)

    # Wilderness and Soil Type Selections
    wilderness_area = st.sidebar.selectbox('Select Wilderness Area', ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4'])
    wilderness_areas = {f'Wilderness_Area{i}': 0 for i in range(1, 5)}
    wilderness_areas[wilderness_area] = 1

    soil_type = st.sidebar.selectbox('Select Soil Type', [f'Soil_Type{i}' for i in range(1, 41) if i not in [7, 15]])
    soil_types = {f'Soil_Type{i}': 0 for i in range(1, 41)}
    soil_types[soil_type] = 1

    # Collect all inputs
    data = {
        'Elevation': elevation,
        'Aspect': aspect,
        'Slope': slope,
        'Horizontal_Distance_To_Hydrology': horizontal_distance_hydrology,
        'Vertical_Distance_To_Hydrology': vertical_distance_hydrology,
        'Horizontal_Distance_To_Roadways': horizontal_distance_roadways,
        'Horizontal_Distance_To_Fire_Points': horizontal_distance_fire_points,
        'Hillshade_9am': hillshade_9am,
        'Hillshade_Noon': hillshade_noon,
        'Hillshade_3pm': hillshade_3pm
    }
    data.update(wilderness_areas)
    data.update(soil_types)
    return pd.DataFrame(data, index=[0])

# Prediction page
if options == "Predict Forest Cover":

    # App title
    st.title('🌲 Forest Cover Type Prediction App')
    st.write("This app predicts the forest cover type based on user input features. Adjust the sliders and inputs to see how the prediction changes.")

        
    # Set some CSS custome animations:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        body {
            font-family: 'Roboto', sans-serif;
            background-image: url("https://images.unsplash.com/photo-1557683304-673a23048d34");
            background-size: cover;
            color: #2E2E2E;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
            animation: fadeIn 1s ease-in-out;
        }
        h1, h2, h3 {
            color: #2E8B57;
            font-weight: 700;
        }
        h1 {
            animation: slideIn 1s ease-out;
        }
        .stButton button {
            background-color: #2E8B57 !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 8px !important;
            transition: background-color 0.3s ease;
            padding: 10px 24px !important;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton button:hover {
            background-color: #267045 !important;
        }
        .css-1v0mbdj img {
            border-radius: 10px;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.3);
            animation: fadeIn 1.5s ease-in-out;
        }
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        @keyframes slideIn {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(0); }
        }
        </style>
        """, unsafe_allow_html=True)

    # Load the trained model
    model = pickle.load(open('forest_cover_model.pkl', 'rb'))

    # Initialize the FeatureEngineer class
    fe = FeatureEngineer()

    # Sidebar input features
    st.sidebar.header('User Input Features')

    def user_input_features():
        # Continuous features
        elevation = st.sidebar.slider('Elevation', 0, 5000, 2500)
        aspect = st.sidebar.slider('Aspect', 0, 360, 180)
        slope = st.sidebar.slider('Slope', 0, 90, 10)
        horizontal_distance_hydrology = st.sidebar.slider('Horizontal Distance to Hydrology', 0, 500, 250)
        vertical_distance_hydrology = st.sidebar.slider('Vertical Distance to Hydrology', 0, 500, 50)
        horizontal_distance_roadways = st.sidebar.slider('Horizontal Distance to Roadways', 0, 10000, 500)
        horizontal_distance_fire_points = st.sidebar.slider('Horizontal Distance to Fire Points', 0, 10000, 4113)
        hillshade_9am = st.sidebar.slider('Hillshade at 9 AM', 0, 255, 200)
        hillshade_noon = st.sidebar.slider('Hillshade at Noon', 0, 255, 220)
        hillshade_3pm = st.sidebar.slider('Hillshade at 3 PM', 0, 255, 150)

        # Wilderness area selection
        wilderness_area = st.sidebar.selectbox(
            'Select Wilderness Area',
            ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']
        )

        # One-hot encode wilderness area
        wilderness_areas = {f'Wilderness_Area{i}': 0 for i in range(1, 5)}
        wilderness_areas[wilderness_area] = 1

        # Soil type selection (excluding Soil_Type7 and Soil_Type15)
        soil_type = st.sidebar.selectbox(
            'Select Soil Type',
            [f'Soil_Type{i}' for i in range(1, 41) if i not in [7, 15]]
        )

        # One-hot encode soil type
        soil_types = {f'Soil_Type{i}': 0 for i in range(1, 41)}
        soil_types[soil_type] = 1

        # Collect the user input into a dataframe
        data = {
            'Elevation': elevation,
            'Aspect': aspect,
            'Slope': slope,
            'Horizontal_Distance_To_Hydrology': horizontal_distance_hydrology,
            'Vertical_Distance_To_Hydrology': vertical_distance_hydrology,
            'Horizontal_Distance_To_Roadways': horizontal_distance_roadways,
            'Horizontal_Distance_To_Fire_Points': horizontal_distance_fire_points,
            'Hillshade_9am': hillshade_9am,
            'Hillshade_Noon': hillshade_noon,
            'Hillshade_3pm': hillshade_3pm
        }

        # Add wilderness areas and soil types to the data
        data.update(wilderness_areas)
        data.update(soil_types)

        # Convert to a DataFrame
        features = pd.DataFrame(data, index=[0])
        return features

    # Collect user input
    input_df = user_input_features()

    # Apply feature engineering before PCA
    X_train = input_df.copy()
    X_train = fe.add_distance_and_hillshade_features(X_train)
    X_train = fe.add_advanced_features(X_train)
    X_train = fe.add_more_features(X_train)

    # Remove Soil_Type7 and Soil_Type15 if they exist
    X_train = X_train.drop(['Soil_Type7', 'Soil_Type15'], axis=1, errors='ignore')

    # Add pca0 and pca1 (binary 0 or 1)
    X_train['pca0'] = np.random.choice([0, 1], len(X_train))
    X_train['pca1'] = np.random.choice([0, 1], len(X_train))

    # Reorder columns to match the training order
    column_order = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
                    'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
                    'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
                    'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type8',
                    'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type16',
                    'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
                    'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
                    'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37',
                    'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'pca0', 'pca1', 'Distance_to_Hydrology', 'Hillshade_mean',
                    'Hillshade_std', 'Horizontal_Distance_combination_HR_1', 'Horizontal_Distance_combination_HR_2',
                    'Horizontal_Distance_combination_HR_3', 'Horizontal_Distance_combination_HR_4',
                    'Horizontal_Distance_combination_HF_1', 'Horizontal_Distance_combination_HF_2',
                    'Horizontal_Distance_combination_HF_3', 'Horizontal_Distance_combination_HF_4',
                    'Horizontal_Distance_combination_RF_1', 'Horizontal_Distance_combination_RF_2',
                    'Horizontal_Distance_combination_RF_3', 'Horizontal_Distance_combination_RF_4',
                    'Horizontal_Distance_mean', '3D_Distance_log_feature_EFR', 'Sqrt_Elevation', 'Adjusted_HDH',
                    'Adjusted_VDH', 'Adjusted_HDR', 'Decayed_Horizontal_Distance_To_Fire_Points',
                    'Decayed_Horizontal_Distance_To_Roadways', 'soil_type_count', 'wilderness_area_count']

    X_train = X_train.reindex(columns=column_order, fill_value=0)

    # Button to trigger prediction
    if st.button('🔍 Predict Cover Type'):
        with st.spinner('✨ Predicting...'):
            # Make predictions
            prediction = model.predict(X_train)[0] + 1

        # Map Forest Cover Type to labels, description, and images
        cover_type_info = {
            1: {"name": "Spruce/Fir", "description": "A type of forest dominated by spruce and fir trees.", "image": "images/spruce_fir.jpg"},
            2: {"name": "Lodgepole Pine", "description": "Found in colder climates, especially in high altitudes.", "image": "images/lodgepole_pine.jpg"},
            3: {"name": "Ponderosa Pine", "description": "Widely distributed in the western United States.", "image": "images/ponderosa_pine.jpg"},
            4: {"name": "Cottonwood/Willow", "description": "Found in riparian zones near water bodies.", "image": "images/cottonwood_willow.jpg"},
            5: {"name": "Aspen", "description": "Aspens are known for their white bark and trembling leaves.", "image": "images/aspen.jpg"},
            6: {"name": "Douglas-fir", "description": "A large, coniferous tree that is important for timber.", "image": "images/douglas_fir.jpg"},
            7: {"name": "Krummholz", "description": "A stunted forest found at the treeline in mountainous regions.", "image": "images/krummholz.jpg"}
        }

        # Display the predicted cover type with description and image
        st.subheader('🌿 Prediction Result')
        cover_info = cover_type_info.get(prediction)
        st.markdown(f'**🌲 The predicted Forest Cover Type is: {cover_info["name"]}**')

        # Display the description
        st.write(cover_info["description"])

        # Display the image (replace the path with the actual location of your images)
        st.image(cover_info["image"], caption=cover_info["name"], use_column_width=True)
        
# Data Analytics page
# Data Analytics page
elif options == "Data Analytics":
    st.title("📊 Data Analytics Dashboard")
    st.write("Explore the dataset through visualizations and statistical summaries.")
    
    # Path to your dataset
    data_path = "data/forest_data.csv"
    
    # Load the dataset
    df = pd.read_csv(data_path)

    # Dataset Overview
    st.write("## 1. Dataset Overview")
    st.write("Shape of the dataset:", df.shape)
    
    # Display the first few records of the dataset
    st.write("### Dataset Sample")
    st.dataframe(df.head())

    # Statistical Summary
    st.write("### 2. Statistical Summary")
    st.write(df.describe())

    # Check for missing values
    st.write("### 3. Missing Values Check")
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        st.write("No missing values detected.")
    else:
        st.write(f"Missing values detected: {missing_values}")
        st.write(df.isnull().sum())

    # Label distribution
    st.write("### 4. Cover Type Distribution")
    cover_type_dist = df["Cover_Type"].value_counts()
    st.bar_chart(cover_type_dist)
    st.write("The dataset is balanced with equal records for each Cover Type.")

    # Visualizations - Buttons to choose different graphs
    st.write("### 5. Visualizations")
    
    continuous_features = ['Elevation', 'Slope', 'Vertical_Distance_To_Hydrology', 
                               'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
                               'Horizontal_Distance_To_Fire_Points', 'Hillshade_9am', 'Hillshade_Noon', 
                               'Hillshade_3pm']
    
    if st.button("Show Boxplots for Continuous Features by Cover Type"):
        st.subheader("Boxplots for Continuous Features by Cover Type")
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        
        for i, feature in enumerate(continuous_features):
            sns.boxplot(x='Cover_Type', y=feature, data=df, ax=axes[i//3, i%3], palette="Set2")
            axes[i//3, i%3].set_title(f'Cover Type vs {feature}')
        plt.tight_layout()
        st.pyplot(fig)

    if st.button("Show Histograms for Continuous Features"):
        st.subheader("Histograms for Continuous Features")
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        colors = ['olivedrab', 'c', 'forestgreen', 'darkcyan', 'olive', 'teal', 'coral', 'greenyellow', 'darkorange']
        for i, feature in enumerate(continuous_features):
            sns.histplot(df[feature], color=colors[i], kde=True, ax=axes[i//3, i%3])
            axes[i//3, i%3].set_title(f'Distribution of {feature}')
        plt.tight_layout()
        st.pyplot(fig)

    if st.button("Show Correlation Heatmap (Excluding Soil Types)"):
        st.subheader("Correlation Heatmap (Excluding Soil Types)")
        # Filter out 'Soil_Type' columns
        df_no_soil = df.loc[:, ~df.columns.str.contains('Soil_Type')]
        corrmat = df_no_soil.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corrmat, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    if st.button("Show Scatter Plot: Aspect vs Hillshade Features"):
        st.subheader("Scatter Plot: Aspect vs Hillshade Features")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        scatter_features = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
        for i, feature in enumerate(scatter_features):
            sns.scatterplot(x='Aspect', y=feature, hue='Cover_Type', data=df, ax=axes[i], palette='viridis', s=50)
            axes[i].set_title(f'Aspect vs {feature}')
        plt.tight_layout()
        st.pyplot(fig)

    if st.button("Show Soil Type Distribution"):
        st.subheader("Soil Type Distribution")
        soil_dummies = df.loc[:, df.columns.str.startswith('Soil_Type')]
        df['Soil'] = soil_dummies.idxmax(axis=1)  # Get the soil type with max value
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.countplot(x='Soil', data=df, palette='Set2', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        st.pyplot(fig)
        st.write("Soil types 7 and 15 are not present in the data and have been removed.")


# Model Evaluation page
elif options == "Model Evaluation":
    st.title("📈 Model Evaluation")
    st.write("Evaluate the model performance and metrics below:")

    # Custom CSS for styling
    st.markdown("""
        <style>
        .stButton button {
            background-color: #2E8B57 !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 10px !important;
            padding: 12px 24px !important;
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        .stButton button:hover {
            background-color: #267045 !important;
            transform: scale(1.05);
        }
        .metric-box {
            background-color: #f0f8f5;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Accuracy button and display
    accuracy = 0.87  # Actual model accuracy

    if st.button("🎯 Show Model Accuracy"):
        st.subheader("✅ Model Accuracy")
        st.markdown(f"""
            <div class="metric-box">
                <h2 style="color: #2E8B57;">The model achieved an accuracy of <strong>{accuracy * 100:.2f}%</strong>.</h2>
                <p>Great job! The model performs well on the dataset.</p>
            </div>
        """, unsafe_allow_html=True)

    # Confusion Matrix button and display
    if st.button("🔍 Show Confusion Matrix"):
        st.subheader("📊 Confusion Matrix")
        confusion_matrix_path = "images/confusion_matrix.png"  # Path to your confusion matrix image
        st.image(confusion_matrix_path, caption="Confusion Matrix", use_column_width=True)
        st.write("The confusion matrix above shows the detailed performance of the model across all classes.")

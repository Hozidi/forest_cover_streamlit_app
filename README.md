
# Forest Cover Type Classifier 🌲🌳

![App Demo](images/forest_cover_app_demo.gif)

This project is a **machine learning model** created as part of an assignment for the **Machine Learning 2 (ML2)** course on Kaggle. The goal of the project is to **classify forest cover types** based on cartographic variables, using data from the U.S. Geological Survey (USGS) and the U.S. Forest Service (USFS). The classifier predicts the type of forest cover for 30x30 meter cells in Colorado’s Roosevelt National Forest. This study area includes four wilderness areas, each with unique characteristics in terms of elevation, tree species, and forest type.

## Project Description 📚

The dataset consists of **quantitative** and **qualitative** variables representing the topographic and ecological conditions of the wilderness areas. The machine learning model uses this data to predict the **forest cover type**, which ranges from **Spruce/Fir** to **Ponderosa Pine** and other forest types.

#### Dataset Attributes:
- **Elevation** (meters)
- **Aspect** (degrees azimuth)
- **Slope** (degrees)
- **Horizontal Distance to Hydrology** (meters)
- **Vertical Distance to Hydrology** (meters)
- **Horizontal Distance to Roadways** (meters)
- **Hillshade Index** at 9am, Noon, and 3pm
- **Horizontal Distance to Fire Points** (meters)
- **Wilderness Areas** (4 binary columns)
- **Soil Types** (40 binary columns)

The **forest cover types** are classified into 7 categories:
1. **Spruce/Fir**
2. **Lodgepole Pine**
3. **Ponderosa Pine**
4. **Cottonwood/Willow**
5. **Aspen**
6. **Douglas-fir**
7. **Krummholz**

---

## Features 🔍

- **Interactive Data Visualization**: Explore the distribution of key environmental factors (e.g., elevation, aspect, slope) with plots and histograms.
- **Machine Learning Predictions**: Classify forest cover types based on input features or uploaded datasets.
- **Real-time Model Feedback**: Understand which features contribute most to the classification.
- **Data Upload**: Upload custom datasets for predictions.

---

## Getting Started 💻

### Prerequisites
- **Python 3.9+**
- **Docker** (optional, for running the app in a container)

### Local Setup 🏃‍♂️

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/forest-cover-classifier.git
   cd forest-cover-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

4. Open the app in your browser at `http://localhost:8501`.

### Docker Setup 🐳

1. Build the Docker image:
   ```bash
   docker build -t forest-cover-app .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8501:8501 forest-cover-app
   ```

---

## Data Description 📝

### Overview

The dataset includes data derived from **U.S. Geological Survey (USGS)** and **U.S. Forest Service (USFS)** resources. The forest cover types are classified based on a variety of cartographic and ecological factors, with a special emphasis on **elevation**, **slope**, and **soil type**. The goal is to predict the forest cover type for a 30x30 meter cell based on these attributes.

### Key Variables

- **Elevation** (meters): Height above sea level.
- **Aspect** (degrees azimuth): Compass direction that a slope faces.
- **Slope** (degrees): The steepness or incline of the terrain.
- **Horizontal/Vertical Distance to Hydrology** (meters): Proximity to the nearest water body.
- **Horizontal Distance to Roadways** (meters): Proximity to the nearest roadway.
- **Hillshade Index** (0-255): Shading effects based on the angle of the sun at different times of day.
- **Horizontal Distance to Fire Points** (meters): Distance to the nearest wildfire ignition point.
- **Wilderness Areas** (binary): Indicating which of the four wilderness areas the observation belongs to.
- **Soil Types** (40 binary columns): Representing different soil types in the study area.
  
The **forest cover types** are:
1. **Spruce/Fir**
2. **Lodgepole Pine**
3. **Ponderosa Pine**
4. **Cottonwood/Willow**
5. **Aspen**
6. **Douglas-fir**
7. **Krummholz**

### Wilderness Areas:

1. **Rawah Wilderness Area**
2. **Neota Wilderness Area**
3. **Comanche Peak Wilderness Area**
4. **Cache la Poudre Wilderness Area**

---

## Technologies Used ⚙️

- **Python**: Used for data manipulation, model training, and app development.
- **Streamlit**: Framework used to create the interactive app.
- **scikit-learn**: For building and training the machine learning model.
- **pandas**: For data manipulation.
- **Docker**: For containerizing the app.

---

## Project Structure 🗂️

```plaintext
├── Dockerfile                 # Docker configuration for containerization
├── README.md                  # Documentation for the project
├── requirements.txt           # Python dependencies
├── streamlit_app.py           # Streamlit app code
├── models/                    # Folder for storing trained models (optional)
├── images/                    # Folder for storing GIF and other images
└── data/                      # Folder for data (optional)
```

---

## Contributing 🤝

Contributions are welcome! If you encounter any issues or want to improve the app, feel free to open a pull request or raise an issue.

---

## License 📄

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Made with hate by [Hocine ZIDI](https://github.com/Hozidi)


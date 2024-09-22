import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

class FeatureEngineer:
    def __init__(self):
        """
        Initializes the FeatureEngineer class.
        """
        pass

    def apply_pca(self, X, pca_model):
        """
        Applies PCA on a single sample or dataset using a pre-fitted PCA model.

        Parameters:
        - X: DataFrame, the input data.
        - pca_model: PCA model, the pre-fitted PCA model to use for transformation.

        Returns:
        - X_pca: DataFrame, the data after PCA transformation.
        """
        pca_trans = pca_model.transform(X)
        pca_columns = [f"pca{i}" for i in range(pca_trans.shape[1])]
        X_pca = pd.DataFrame(pca_trans, columns=pca_columns)

        return X_pca

    def add_distance_and_hillshade_features(self, all_data):
        """
        Adds engineered features related to distances and hillshade to the dataset.
        """
        # Distance to Hydrology
        all_data["Distance_to_Hydrology"] = np.sqrt(np.square(all_data["Vertical_Distance_To_Hydrology"]) +
                                                     np.square(all_data["Horizontal_Distance_To_Hydrology"]))

        # Hillshade metrics
        hillshade_cols = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
        all_data["Hillshade_mean"] = all_data[hillshade_cols].mean(axis=1)
        all_data["Hillshade_std"] = all_data[hillshade_cols].std(axis=1)

        # Horizontal distance combinations
        cols = ["Horizontal_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"]
        names = ["H", "R", "F"]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                all_data[f"Horizontal_Distance_combination_{names[i]}{names[j]}_1"] = all_data[cols[i]] + all_data[cols[j]]
                all_data[f"Horizontal_Distance_combination_{names[i]}{names[j]}_2"] = (all_data[cols[i]] + all_data[cols[j]]) / 2
                all_data[f"Horizontal_Distance_combination_{names[i]}{names[j]}_3"] = all_data[cols[i]] - all_data[cols[j]]
                all_data[f"Horizontal_Distance_combination_{names[i]}{names[j]}_4"] = np.abs(all_data[cols[i]] - all_data[cols[j]])

        # Mean of horizontal distances
        all_data["Horizontal_Distance_mean"] = all_data[cols].mean(axis=1)

        return all_data

    def add_advanced_features(self, all_data):
        """
        Adds advanced features including log-based features, square root transformations,
        adjusted distances, and decay functions applied to certain distances.
        """
        all_data['3D_Distance_log_feature_EFR'] = np.sqrt(np.log(all_data['Elevation']+1)**2 +
                                                          np.log(all_data['Horizontal_Distance_To_Fire_Points']+1)**2 +
                                                          np.log(all_data['Horizontal_Distance_To_Roadways']+1)**2)

        all_data['Sqrt_Elevation'] = np.sqrt(all_data['Elevation'])

        # Adjusted distances
        all_data['Adjusted_HDH'] = all_data['Elevation'] - 0.17 * all_data['Horizontal_Distance_To_Hydrology']
        all_data['Adjusted_VDH'] = all_data['Elevation'] - 0.9 * all_data['Vertical_Distance_To_Hydrology']
        all_data['Adjusted_HDR'] = all_data['Elevation'] - 0.03 * all_data['Horizontal_Distance_To_Roadways']

        # Decay function for distance
        def decay_function(distance, half_life):
            return np.exp(-distance / half_life)

        all_data['Decayed_Horizontal_Distance_To_Fire_Points'] = decay_function(all_data['Horizontal_Distance_To_Fire_Points'], 200)
        all_data['Decayed_Horizontal_Distance_To_Roadways'] = decay_function(all_data['Horizontal_Distance_To_Roadways'], 150)

        return all_data

    def add_more_features(self, all_data):
        """
        Adds additional engineered features for soil types, wilderness areas, and combinations.
        """
        if 'Soil_Type7' not in all_data.columns:
            all_data['Soil_Type7'] = 0
        if 'Soil_Type15' not in all_data.columns:
            all_data['Soil_Type15'] = 0

        # Counting soil types and wilderness areas
        soil_features = [x for x in all_data.columns if x.startswith("Soil_Type")]
        wilderness_features = [x for x in all_data.columns if x.startswith("Wilderness_Area")]
        all_data["soil_type_count"] = all_data[soil_features].sum(axis=1)
        all_data["wilderness_area_count"] = all_data[wilderness_features].sum(axis=1)

        return all_data

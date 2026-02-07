import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class HybridAlgaeModel:
    """
    The Core 'Hybrid' Predictive Engine.
    Combines Unsupervised Clustering (K-Means) with Supervised Classification (Random Forest).
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def generate_synthetic_data(self, n_samples=2000):
        """
        Simulates realistic sensor data for training.
        """
        np.random.seed(42)
        data = {
            'temperature': np.random.normal(25, 3, n_samples),
            'pH': np.random.normal(8.1, 0.2, n_samples),
            'turbidity': np.random.normal(10, 5, n_samples),
            'salinity': np.random.normal(35, 2, n_samples),
            'nitrate': np.random.exponential(2.0, n_samples), # Agricultural runoff skew
            'phosphate': np.random.exponential(0.5, n_samples),
            'dissolved_oxygen': np.random.normal(6, 1.5, n_samples),
            'light_intensity': np.random.normal(1200, 300, n_samples) # New Feature V5.0
        }
        df = pd.DataFrame(data)
        
        # Artificial Rule for Labeling
        # High Temp + High Nutrients + High Light = Bloom
        df['bloom_risk'] = 0 # Low
        condition_med = (df['temperature'] > 26) & (df['nitrate'] > 2)
        # Light boosts bloom risk significantly
        condition_high = (df['temperature'] > 28) & (df['nitrate'] > 4) & (df['phosphate'] > 1) & (df['light_intensity'] > 1500)
        
        df.loc[condition_med, 'bloom_risk'] = 1 # Medium
        df.loc[condition_high, 'bloom_risk'] = 2 # Critical
        
        return df

    def train(self):
        """
        Executes the Hybrid Training Pipeline.
        """
        df = self.generate_synthetic_data()
        X = df.drop('bloom_risk', axis=1)
        y = df['bloom_risk']
        
        # Step 1: K-Means Clustering (Unsupervised Feature Engineering)
        # We add the Cluster ID as a feature to help the Random Forest
        X_scaled = self.scaler.fit_transform(X)
        clusters = self.kmeans.fit_predict(X_scaled)
        X['cluster_id'] = clusters
        
        # Step 2: Random Forest (Supervised Classification)
        self.rf_classifier.fit(X, y)
        self.is_trained = True
        return self.rf_classifier.feature_importances_, X.columns

    def predict(self, input_features):
        """
        Predicts risk for a single sensor reading.
        features: dict of {temp, pH, turbidity, salinity, nitrate, phosphate, DO}
        """
        if not self.is_trained:
            self.train()
            
        df = pd.DataFrame([input_features])
        
        # 1. Cluster it
        scaled = self.scaler.transform(df)
        cluster = self.kmeans.predict(scaled)[0]
        df['cluster_id'] = cluster
        
        # 2. Classify it
        risk_class = self.rf_classifier.predict(df)[0]
        probabilities = self.rf_classifier.predict_proba(df)[0]
        confidence = max(probabilities) * 100
        
        risk_labels = {0: "Low Risk", 1: "Moderate", 2: "CRITICAL BLOOM"}
        return risk_labels[risk_class], confidence, cluster

    def get_feature_importance(self):
        if not self.is_trained:
            self.train()
        return dict(zip(self.rf_classifier.feature_names_in_, self.rf_classifier.feature_importances_))

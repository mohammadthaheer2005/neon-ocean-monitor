import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class DataMiningOrchestrator:
    """
    The Intelligence Unit.
    Extracts hidden patterns, clusters, and anomalies from Ocean Data.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)

    def perform_clustering(self, df):
        """
        Groups ocean pixels into 3 Ecological Zones:
        0: Low Risk / Safe
        1: Moderate / Warning
        2: High Risk / Bloom Critical
        """
        # We use SST and Chlorophyll (if available, else simulated) for clustering
        features = df[['sst']].copy()
        
        # Simulate Chlorophyll if not present (for demo robustness)
        if 'chlorophyll' not in df.columns:
            # Correlation: Higher Temp often correlates with blooms (simplified)
            features['chlorophyll'] = (features['sst'] * 0.5) + np.random.normal(0, 0.5, len(df))
            
        scaled_features = self.scaler.fit_transform(features)
        
        # Train K-Means
        clusters = self.kmeans.fit_predict(scaled_features)
        df['cluster'] = clusters
        
        # Label the clusters based on mean SST (Assuming warmer = riskier for this demo)
        cluster_means = df.groupby('cluster')['sst'].mean().sort_values()
        risk_map = {cluster_means.index[0]: 'Safe', cluster_means.index[1]: 'Warning', cluster_means.index[2]: 'Critical'}
        df['risk_label'] = df['cluster'].map(risk_map)
        
        return df

    def detect_anomalies(self, df):
        """
        Identifies "Black Swan" events (outliers).
        """
        features = df[['sst']].values
        # -1 = Anomaly, 1 = Normal
        outliers = self.anomaly_detector.fit_predict(features)
        df['is_anomaly'] = outliers == -1
        return df

    def get_correlations(self, df):
        """
        Returns correlation matrix between variables.
        """
        # adding synthetic chlorophyll for correlation demo
        if 'chlorophyll' not in df.columns:
            df['chlorophyll'] = (df['sst'] * 0.5) + np.random.normal(0, 0.5, len(df))
            
        return df[['sst', 'chlorophyll']].corr()

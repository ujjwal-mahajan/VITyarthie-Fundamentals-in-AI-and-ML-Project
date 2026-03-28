# VITyarthie-Fundamentals-in-AI-and-ML-Project
StudyBuddy AI is a machine learning-based tool designed to solve the "Silo Effect" in large academic environments like VIT. In a competitive engineering landscape, students often struggle to find peers who have complementary skill sets (e.g., a strong coder looking for a partner strong in Mathematics).

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1.Create a Synthetic Dataset (Simulating VIT Student Data)
# Scores are from 1-10 (1: Needs Help, 10: Expert)
data = {
    'Student_ID': [f'26CSE{i:03d}' for i in range(1, 21)],
    'Math_Score': [9, 3, 8, 4, 10, 2, 7, 5, 9, 3, 8, 2, 6, 4, 10, 1, 7, 5, 8, 4],
    'Coding_Score': [4, 10, 3, 9, 2, 8, 5, 7, 3, 10, 4, 9, 6, 8, 1, 10, 5, 7, 4, 9],
    'Theory_Score': [7, 6, 8, 5, 7, 6, 9, 4, 8, 5, 7, 6, 10, 5, 8, 4, 7, 6, 9, 5]
}

df = pd.DataFrame(data)

# 2.Preprocessing
# We only use the scores for clustering, not the ID
X = df[['Math_Score', 'Coding_Score', 'Theory_Score']]

# Scaling data (Standard Practice in ML Fundamentals)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3.Building the Model (K-Means)
# We will create 4 clusters (Study Groups)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Study_Group'] = kmeans.fit_predict(X_scaled)

# 4.Results Output
print("--- Study Buddy Assignments ---")
print(df[['Student_ID', 'Study_Group']].head(10))

# 5.Simple Recommendation Logic
def find_my_buddies(student_id):
    if student_id not in df['Student_ID'].values:
        return "Student ID not found."
    
group = df[df['Student_ID'] == student_id]['Study_Group'].values[0]
buddies = df[(df['Study_Group'] == group) & (df['Student_ID'] != student_id)]
    
print(f"\nRecommendations for {student_id}:")
print(f"You have been assigned to Group {group}.")
print("Your potential study partners are:")
return buddies[['Student_ID', 'Math_Score', 'Coding_Score', 'Theory_Score']]

# Example Run
print(find_my_buddies('26CSE001'))

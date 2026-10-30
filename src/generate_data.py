
import numpy as np
import pandas as pd
from pathlib import Path

# Reproducibility
np.random.seed(42)

# Number of students
n_students = 1000

# Generate student features
attendance = np.random.uniform(40, 100, n_students)
assessment_score = np.random.uniform(30, 100, n_students)
study_hours = np.random.uniform(1, 8, n_students)
assignment_completion = np.random.uniform(30, 100, n_students)
previous_score = np.random.uniform(30, 100, n_students)

# Generate target variable
final_score = (
    0.20 * attendance
    + 0.35 * assessment_score
    + 1.5 * study_hours
    + 0.15 * assignment_completion
    + 0.20 * previous_score
    + np.random.normal(0, 5, n_students)
)

# Keep scores between 0 and 100
final_score = np.clip(final_score, 0, 100)

# Create DataFrame
df = pd.DataFrame({
    "attendance": attendance.round(2),
    "assessment_score": assessment_score.round(2),
    "study_hours": study_hours.round(2),
    "assignment_completion": assignment_completion.round(2),
    "previous_score": previous_score.round(2),
    "final_score": final_score.round(2)
})

# Save dataset
output_path = Path("data/student_data.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print("Dataset generated successfully!")
print("Shape:", df.shape)
print("\nFirst five records:")
print(df.head())

print("\nDataset statistics:")
print(df.describe())

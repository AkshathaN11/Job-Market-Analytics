import pandas as pd

df = pd.read_csv("data_jobs.csv/NaukriData_Data Science.csv")

df.columns = df.columns.str.strip()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

df = df.drop_duplicates()

# Top hiring companies
top_companies = df["Company_Names"].value_counts().head(10)

print("\nTop 10 Hiring Companies:")
print(top_companies)
# Top hiring locations
top_locations = df["Locations"].value_counts().head(10)

print("\nTop 10 Hiring Locations:")
print(top_locations)
# In-demand skills
skills = [
    "Python", "SQL", "Excel", "Power BI", "Tableau",
    "Machine Learning", "Deep Learning", "NLP",
    "TensorFlow", "PyTorch", "AWS", "Azure",
    "Statistics", "Data Visualization", "Pandas",
    "NumPy", "Spark"
]

skill_counts = {}

for skill in skills:
    count = df["Skills"].str.contains(skill, case=False, na=False).sum()
    skill_counts[skill] = count

top_skills = pd.Series(skill_counts).sort_values(ascending=False)

print("\nMost In-Demand Skills:")
print(top_skills)
import re

# Extract salary values
def get_salary(text):
    numbers = re.findall(r'\d+(?:\.\d+)?', str(text))
    
    if len(numbers) >= 2:
        return (float(numbers[0]) + float(numbers[1])) / 2
    elif len(numbers) == 1:
        return float(numbers[0])
    return None

df["Average_Salary_LPA"] = df["Package_Details"].apply(get_salary)

print("\nSalary Analysis:")
print("Jobs with salary information:", df["Average_Salary_LPA"].notna().sum())

print("Average Salary:", round(df["Average_Salary_LPA"].mean(), 2), "LPA")
print("Highest Salary:", round(df["Average_Salary_LPA"].max(), 2), "LPA")
# Experience analysis
def get_min_experience(text):
    numbers = re.findall(r'\d+', str(text))
    if numbers:
        return int(numbers[0])
    return None

df["Min_Experience_Years"] = df["Experience_Required"].apply(get_min_experience)

print("\nExperience Analysis:")
print("Average required experience:",
      round(df["Min_Experience_Years"].mean(), 1), "years")

print("\nExperience-wise job count:")
print(df["Min_Experience_Years"].value_counts().sort_index())
import matplotlib.pyplot as plt

# Top job roles chart
top_jobs = df["Job_Titles"].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_jobs.sort_values().plot(kind="barh")

plt.title("Top 10 Most Demanded Job Roles")
plt.xlabel("Number of Job Postings")
plt.ylabel("Job Role")

plt.tight_layout()
plt.savefig("screenshots/top_job_roles.png")
plt.show()
# Top skills chart
top_skills.head(10).sort_values().plot(kind="barh", figsize=(10, 6))

plt.title("Top 10 In-Demand Skills")
plt.xlabel("Number of Job Postings")
plt.ylabel("Skill")

plt.tight_layout()
plt.savefig("screenshots/top_skills.png")
plt.show()
# Top locations chart
top_locations = df["Locations"].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_locations.sort_values().plot(kind="barh")

plt.title("Top 10 Job Locations")
plt.xlabel("Number of Job Postings")
plt.ylabel("Location")

plt.tight_layout()
plt.savefig("screenshots/top_locations.png")
plt.show()
# Salary chart
salary_data = df["Average_Salary_LPA"].dropna()

plt.figure(figsize=(10, 6))
plt.hist(salary_data, bins=10)

plt.title("Salary Distribution")
plt.xlabel("Average Salary (LPA)")
plt.ylabel("Number of Job Postings")

plt.tight_layout()
plt.savefig("screenshots/salary_distribution.png")
plt.show()
# Export cleaned dataset
df.to_csv("data_jobs_cleaned.csv", index=False)

print("\nCleaned dataset exported successfully!")
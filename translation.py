import pandas as pd


INPUT_FILE = "ChatGPT/TS2.csv"
OUTPUT_FILE = "ChatGPT/TS2_translated.csv"


ACTIVITY_MAPPING = {
    "Activity 01": "ER Registration",
    "Activity 02": "ER Triage",
    "Activity 03": "ER Sepsis Triage",
    "Activity 04": "CRP",
    "Activity 05": "Leucocytes",
    "Activity 06": "LacticAcid",
    "Activity 07": "IV Liquid",
    "Activity 08": "IV Antibiotics",
    "Activity 09": "Admission IC",
    "Activity 10": "Admission NC",
    "Activity 11": "Return ER",
    "Activity 12": "Release A",
    "Activity 13": "Release B",
    "Activity 14": "Release C",
    "Activity 15": "Release D",
    "Activity 16": "Release E",
}


def translate_slash_separated_activities(value):
    activities = [
        activity.strip()
        for activity in str(value).split("/")
        if activity.strip()
    ]

    translated = []

    for activity in activities:
        if activity not in ACTIVITY_MAPPING:
            raise ValueError(f"No mapping defined for activity: {activity}")

        translated.append(ACTIVITY_MAPPING[activity])

    return "/".join(translated)


df = pd.read_csv(INPUT_FILE)

required_cols = ["Case ID", "Activity", "Timestamp", "Group", "Lifecycle"]
missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["Activity"] = df["Activity"].apply(translate_slash_separated_activities)

# Optional: normalize lifecycle values
df["Lifecycle"] = df["Lifecycle"].astype(str).str.replace("DEFAULT", "complete")

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved translated log to: {OUTPUT_FILE}")
print(df.head())
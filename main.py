import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================
# LOAD DATASET
# =====================================

data = pd.read_csv("cricket.csv")

# =====================================
# LABEL ENCODING
# =====================================

le_match = LabelEncoder()
le_yesno = LabelEncoder()
le_result = LabelEncoder()

data["MatchType"] = le_match.fit_transform(data["MatchType"])
data["TossWinner"] = le_yesno.fit_transform(data["TossWinner"])
data["Result"] = le_result.fit_transform(data["Result"])

# =====================================
# FEATURES & TARGET
# =====================================

X = data[[
    "MatchType",
    "Innings",
    "CurrentRuns",
    "Wickets",
    "Overs",
    "Target",
    "TossWinner"
]]

y = data["Result"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# TRAIN MODEL
# =====================================

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

# =====================================
# MODEL ACCURACY
# =====================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# =====================================
# USER INTERFACE
# =====================================

print("\n======================================")
print("🏏 Live Cricket Match Prediction System")
print("======================================")

print(f"📊 Model Accuracy: {accuracy * 100:.2f}%")

# =====================================
# MATCH TYPE
# =====================================

print("\nSelect Match Type:")
print("1. T20")
print("2. ODI")
print("3. TEST")

choice = int(input("Enter choice: "))

if choice == 1:
    match_type = "T20"
    max_overs = 20

elif choice == 2:
    match_type = "ODI"
    max_overs = 50

elif choice == 3:
    match_type = "TEST"
    max_overs = 90

else:
    print("❌ Invalid Match Type")
    exit()

# =====================================
# INNINGS
# =====================================

innings = int(input("\nEnter Innings (1 or 2): "))

if innings not in [1, 2]:
    print("❌ Innings must be 1 or 2")
    exit()

# =====================================
# RUNS
# =====================================

runs = int(input("Enter Current Runs: "))

# =====================================
# WICKETS
# =====================================

wickets = int(input("Enter Wickets Lost (0-10): "))

if wickets < 0 or wickets > 10:
    print("❌ Invalid wickets")
    exit()

# =====================================
# OVERS
# =====================================

overs = float(input(f"Enter Overs Completed (0-{max_overs}): "))

if overs < 0 or overs > max_overs:
    print("❌ Invalid overs")
    exit()

# =====================================
# TARGET
# =====================================

if innings == 2:
    target = int(input("Enter Target Score: "))
else:
    target = 0

# =====================================
# TOSS
# =====================================

toss = input("Did team win toss? (Yes/No): ").capitalize()

if toss not in ["Yes", "No"]:
    print("❌ Enter only Yes or No")
    exit()

# =====================================
# CALCULATIONS
# =====================================

if overs > 0:
    current_rr = runs / overs
else:
    current_rr = 0

if innings == 2:

    runs_needed = target - runs
    overs_left = max_overs - overs

    if overs_left > 0:
        required_rr = runs_needed / overs_left
    else:
        required_rr = 0

else:
    runs_needed = 0
    required_rr = 0

# =====================================
# ENCODE INPUTS
# =====================================

match_encoded = le_match.transform([match_type])[0]
toss_encoded = le_yesno.transform([toss])[0]

# =====================================
# CREATE INPUT DATAFRAME
# =====================================

new_data = pd.DataFrame([[
    match_encoded,
    innings,
    runs,
    wickets,
    overs,
    target,
    toss_encoded
]], columns=[
    "MatchType",
    "Innings",
    "CurrentRuns",
    "Wickets",
    "Overs",
    "Target",
    "TossWinner"
])

# =====================================
# PREDICTION
# =====================================

prediction = model.predict(new_data)

result = le_result.inverse_transform(prediction)

# =====================================
# FINAL OUTPUT
# =====================================

print("\n======================================")
print("📊 Match Analysis")
print("======================================")

print(f"🏏 Match Type: {match_type}")
print(f"📈 Current Score: {runs}/{wickets}")
print(f"⏱ Overs Completed: {overs}")
print(f"🔥 Current Run Rate: {current_rr:.2f}")

if innings == 2:
    print(f"🎯 Target: {target}")
    print(f"🏃 Runs Needed: {runs_needed}")
    print(f"⚡ Required Run Rate: {required_rr:.2f}")

print("\n======================================")
print("🤖 AI Prediction")
print("======================================")

print(f"🏆 Team will likely: {result[0]}")

print("======================================")
from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================
# CREATE FLASK APP
# =====================================

app = Flask(__name__)

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
# HOME ROUTE
# =====================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    current_rr = 0
    required_rr = 0
    runs_needed = 0

    if request.method == "POST":

        # =========================
        # GET FORM DATA
        # =========================

        match_type = request.form["match_type"]
        innings = int(request.form["innings"])
        runs = int(request.form["runs"])
        wickets = int(request.form["wickets"])
        overs = float(request.form["overs"])
        toss = request.form["toss"]

        # =========================
        # MATCH TYPE OVERS
        # =========================

        if match_type == "T20":
            max_overs = 20

        elif match_type == "ODI":
            max_overs = 50

        else:
            max_overs = 90

        # =========================
        # TARGET
        # =========================

        if innings == 2:
            target = int(request.form["target"])
        else:
            target = 0

        # =========================
        # CURRENT RUN RATE
        # =========================

        if overs > 0:
            current_rr = runs / overs

        # =========================
        # CHASE CALCULATIONS
        # =========================

        if innings == 2:

            runs_needed = target - runs
            overs_left = max_overs - overs

            if overs_left > 0:
                required_rr = runs_needed / overs_left

        # =========================
        # ENCODE INPUTS
        # =========================

        match_encoded = le_match.transform([match_type])[0]
        toss_encoded = le_yesno.transform([toss])[0]

        # =========================
        # CREATE DATAFRAME
        # =========================

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

        # =========================
        # PREDICTION
        # =========================

        prediction = model.predict(new_data)

        result = le_result.inverse_transform(prediction)[0]

    # =====================================
    # RETURN HTML PAGE
    # =====================================

    return render_template(
        "index.html",
        result=result,
        accuracy=round(accuracy * 100, 2),
        current_rr=round(current_rr, 2),
        required_rr=round(required_rr, 2),
        runs_needed=runs_needed
    )

# =====================================
# RUN APP
# =====================================

if __name__ == "__main__":
    app.run(debug=True)
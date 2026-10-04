import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="OrbitalGuard AI",
    page_icon="🛰️",
    layout="wide"
)


# =====================================================
# CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 80% 10%,
            rgba(107,39,55,0.30),
            transparent 30%
        ),
        radial-gradient(
            circle at 10% 80%,
            rgba(70,60,100,0.15),
            transparent 30%
        ),
        #080b14;

    color: white;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}


/* ================= HERO ================= */

.hero {
    padding: 40px;
    border-radius: 25px;

    background:
        linear-gradient(
            135deg,
            rgba(107,39,55,0.55),
            rgba(20,24,40,0.95)
        );

    border: 1px solid rgba(255,255,255,0.10);

    position: relative;
    overflow: hidden;

    margin-bottom: 30px;
}

.hero-title {
    font-size: 45px;
    font-weight: 800;
}

.hero-text {
    color: #bfc3d1;
    font-size: 17px;
    max-width: 650px;
    margin-top: 10px;
}

.satellite {
    position: absolute;
    right: 100px;
    top: 55px;

    font-size: 80px;

    animation: float 4s ease-in-out infinite;
}

@keyframes float {

    0% {
        transform: translateY(0px) rotate(-3deg);
    }

    50% {
        transform: translateY(-12px) rotate(3deg);
    }

    100% {
        transform: translateY(0px) rotate(-3deg);
    }

}


/* ================= CARDS ================= */

.card {

    background: rgba(255,255,255,0.05);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 18px;

    padding: 22px;

    min-height: 110px;

}

.card-title {

    color: #9fa5b5;

    font-size: 13px;

    text-transform: uppercase;

    letter-spacing: 1px;

}

.card-value {

    font-size: 30px;

    font-weight: 800;

    margin-top: 8px;

}


/* ================= BUTTON ================= */

.stButton > button {

    width: 100%;

    height: 50px;

    border-radius: 12px;

    border: none;

    background:
        linear-gradient(
            90deg,
            #6B2737,
            #8E4A5A
        );

    color: white;

    font-size: 16px;

    font-weight: 700;

}

.stButton > button:hover {

    box-shadow:
        0 8px 25px
        rgba(107,39,55,0.5);

}


/* ================= PANEL ================= */

.panel {

    padding: 25px;

    border-radius: 20px;

    background:
        rgba(255,255,255,0.04);

    border:
        1px solid
        rgba(255,255,255,0.08);

}


/* ================= FOOTER ================= */

.footer {

    text-align: center;

    color: #777d8c;

    margin-top: 40px;

}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATASET
# =====================================================

data = pd.read_csv("dataset.csv")


# =====================================================
# FEATURES
# =====================================================

features = [

    "sampling",
    "duration",
    "len",
    "mean",
    "var",
    "std",
    "kurtosis",
    "skew",
    "n_peaks",
    "smooth10_n_peaks",
    "smooth20_n_peaks",
    "diff_peaks",
    "diff2_peaks",
    "diff_var",
    "diff2_var",
    "gaps_squared",
    "len_weighted",
    "var_div_duration",
    "var_div_len"

]


# =====================================================
# PREPARE DATA
# =====================================================

X = data[features]

y = data["anomaly"]


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


# =====================================================
# RANDOM FOREST MODEL
# =====================================================

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42

)


model.fit(X_train, y_train)


# =====================================================
# HERO SECTION
# ==================================================
st.markdown("""
<div class="hero">
    <div class="satellite">🛰️</div>
    <div class="hero-title">OrbitalGuard AI</div>
    <div class="hero-text">
        Satellite Telemetry Anomaly Detection powered by Machine Learning.
        Monitor satellite telemetry and identify unusual behaviour.
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# MISSION OVERVIEW
# =====================================================

st.subheader("📡 Mission Overview")


total = len(data)

normal = int(
    (data["anomaly"] == 0).sum()
)

anomalous = int(
    (data["anomaly"] == 1).sum()
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric("Telemetry Segments", f"{total:,}")


with col2:
    st.metric("Normal Signals", f"{normal:,}")


with col3:

    st.metric("Anomalous Signals", f"{anomalous:,}")

# =====================================================
# TELEMETRY ANALYZER
# =====================================================

st.subheader("🔍 Telemetry Analyzer")

st.markdown(
    '<div class="panel">',
    unsafe_allow_html=True
)

row_number = st.number_input(
    "Select telemetry segment",
    min_value=0,
    max_value=len(data) - 1,
    value=78,
    step=1
)

if st.button("🚀 ANALYZE TELEMETRY"):

    selected_row = data.iloc[[row_number]]

    input_data = selected_row[features]

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    probability_percent = probability * 100

    if prediction == 1:

        st.error("🚨 ANOMALY DETECTED!")

        if probability_percent >= 80:
            risk = "🔴 HIGH"

        elif probability_percent >= 50:
            risk = "🟡 MEDIUM"

        else:
            risk = "🟢 LOW"

    else:

        st.success("🟢 TELEMETRY IS NORMAL")

        risk = "🟢 LOW"

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Anomaly Probability",
            f"{probability_percent:.1f}%"
        )

    with col2:
        st.metric(
            "Risk Level",
            risk
        )

    actual = selected_row["anomaly"].iloc[0]

    if actual == 1:
        st.write("Actual dataset label: **ANOMALY**")
    else:
        st.write("Actual dataset label: **NORMAL**")

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# CAN YOU SAVE THE SATELLITE?
# =====================================================

st.subheader("🎮 Can You Save the Satellite?")

st.markdown("""
<div class="panel">

<h3>🚨 SATELLITE EMERGENCY</h3>

<p>
OrbitalGuard AI has detected unusual telemetry behaviour.
The satellite may be experiencing an abnormal condition.
</p>

<p><b>Your mission:</b> Choose the safest action.</p>

</div>
""", unsafe_allow_html=True)


satellite_names = [
    "ORBIT-01",
    "NOVA-02",
    "AURORA-03",
    "HORIZON-04",
    "COSMOS-05",
    "STELLAR-06",
    "PIONEER-07",
    "ZENITH-08",
    "ECLIPSE-09",
    "ASTRA-10",
    "VOYAGER-11",
    "LUMEN-12",
    "NEBULA-13",
    "SOLARIS-14",
    "GALAXY-15",
    "APEX-16",
    "ODYSSEY-17",
    "QUASAR-18",
    "CELESTIA-19",
    "ORION-20",
    "VANGUARD-21",
    "SKYLINE-22",
    "STARDUST-23",
    "PULSAR-24",
    "ECLIPTICA-25",
    "INFINITY-26",
    "ASTRON-27",
    "STARLIGHT-28",
    "NOVA-X29",
    "HORIZON-X30",
    "AURORA-X31",
    "COSMOS-X32",
    "STARLIGHT-X33",
    "SOLAR-X34",
    "DEEPSPACE-35",
    "EXPLORER-36",
    "GALILEO-37",
    "DISCOVERY-38",
    "PHOENIX-39",
    "TRIDENT-40",
    "SIRIUS-41",
    "VEGA-42",
    "LYRA-43",
    "ATLAS-44",
    "GENESIS-45",
    "POLARIS-46",
    "ECHO-47",
    "RADIANT-48",
    "SKYWARD-49",
    "CELESTIAL-50"
]


game_row = st.selectbox(
    "🛰️ Select Satellite",
    range(len(data)),
    format_func=lambda x: (
        f"{satellite_names[x % len(satellite_names)]} "
        f"— Mission {x + 1}"
    ),
    index=0
)

satellite_name = satellite_names[
    game_row % len(satellite_names)
]

st.info(
    f"🛰️ **Satellite:** {satellite_name}  |  "
    f"📡 **Telemetry Segment:** {game_row}"
)


game_input = data.iloc[[game_row]][features]

game_prediction = model.predict(game_input)[0]

game_probability = (
    model.predict_proba(game_input)[0][1] * 100
)


st.metric(
    "AI Anomaly Probability",
    f"{game_probability:.1f}%"
)


st.write("### 🛡️ What should you do?")


choice = st.radio(
    "Choose your action:",
    [
        "🟢 Continue Mission",
        "🟡 Enter Safe Mode",
        "🔴 Shut Down Satellite"
    ]
)


if st.button("🚀 MAKE DECISION"):

    if game_probability >= 80:

        if choice == "🟡 Enter Safe Mode":

            st.success(
                f"✅ Excellent decision! "
                f"{satellite_name} has been placed into Safe Mode."
            )

            st.balloons()

        elif choice == "🔴 Shut Down Satellite":

            st.warning(
                "⚠️ Safe, but more drastic than necessary. "
                "Safe Mode would normally be preferred."
            )

        else:

            st.error(
                f"🚨 Dangerous decision! "
                f"{satellite_name} is showing high-risk behaviour."
            )

    elif game_probability >= 50:

        if choice == "🟡 Enter Safe Mode":

            st.success(
                f"✅ Good decision! "
                f"{satellite_name} is being monitored in Safe Mode."
            )

        elif choice == "🟢 Continue Mission":

            st.success(
                "👍 Reasonable decision, "
                "but monitoring should continue."
            )

        else:

            st.warning(
                "⚠️ Shutdown may be unnecessary "
                "for this situation."
            )

    else:

        if choice == "🟢 Continue Mission":

            st.success(
                f"✅ Correct! {satellite_name} "
                "can safely continue its mission."
            )

        else:

            st.warning(
                "⚠️ This response is probably "
                "more cautious than necessary."
            )


# =====================================================
# AUTOMATIC ANOMALY SCANNER
# =====================================================

st.subheader("🚨 Automatic Anomaly Scanner")

if st.button("🔎 SCAN ALL TELEMETRY"):

    predictions = model.predict(X)

    actual_anomalies = int(
        (y == 1).sum()
    )

    correct_anomalies = int(
        ((y == 1) & (predictions == 1)).sum()
    )

    missed_anomalies = int(
        ((y == 1) & (predictions == 0)).sum()
    )

    false_alarms = int(
        ((y == 0) & (predictions == 1)).sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Actual Anomalies",
            actual_anomalies
        )

    with col2:
        st.metric(
            "Detected Correctly",
            correct_anomalies
        )

    with col3:
        st.metric(
            "Missed Anomalies",
            missed_anomalies
        )

    with col4:
        st.metric(
            "False Alarms",
            false_alarms
        )


    

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

st.subheader(
    "📊 Important Telemetry Features"
)


importance = model.feature_importances_


feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance": importance

})


feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)
st.bar_chart(
    feature_importance
    .head(10)
    .set_index("Feature"),
    height=400
)


# =====================================================
# ABOUT
# =====================================================

st.subheader(
    "🛰️ About OrbitalGuard AI"
)


st.markdown("""

<div class="panel">

<b>OrbitalGuard AI</b> uses a Random Forest
Machine Learning model to classify satellite
telemetry as <b>Normal</b> or <b>Anomalous</b>.

<br><br>

<b>Machine Learning Model:</b> Random Forest
<br>

<b>Dataset:</b> OPSSAT-AD
<br>

<b>Test Accuracy:</b> Approximately 95%
<br>

<b>Input:</b> Satellite telemetry features

</div>

""", unsafe_allow_html=True)


# =====================================================
# FOOTER
# =====================================================

st.markdown("""

<div class="footer">

🛰️ OrbitalGuard AI

<br>

Satellite Telemetry Anomaly Detection

<br>

Built with Python + Machine Learning

</div>

""", unsafe_allow_html=True)
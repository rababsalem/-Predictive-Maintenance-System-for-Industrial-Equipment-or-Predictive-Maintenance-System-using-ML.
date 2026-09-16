# Predictive-Maintenance
#  Industrial Predictive Maintenance System

A full-stack Machine Learning and Data Engineering pipeline designed to predict the **Remaining Useful Life (RUL)** of turbofan jet engines using sensor degradation data from the **NASA CMAPSS Dataset**.

This project simulates real-world predictive maintenance strategies used in heavy industry and oil & gas infrastructure to minimize unplanned downtime and optimize component replacement schedules.

---

##  Project Architecture & Workflow

1. **Data Preprocessing & Feature Engineering:**
   - Ingested time-series sensor telemetry data (`FD001`).
   - Calculated exact ground-truth Remaining Useful Life (RUL) per engine cycle.
   - Applied **Piecewise Linear Target Clipping** (capped RUL at 125 cycles) based on domain research to eliminate early-life noise.
   - Dropped invariant sensors and normalized high-dimensional features using `StandardScaler`.

2. **Model Training & Validation:**
   - Implemented an **Engine-Level Split** to avoid data leakage between training and evaluation cycles.
   - Trained a **Random Forest Regressor** to capture nonlinear sensor degradation trajectories.
   - Achieved high predictive accuracy:
     - **Mean Absolute Error (MAE):** `12.37 cycles`
     - **Root Mean Squared Error (RMSE):** `17.10 cycles`
     - **R² Score:** `0.832`

3. **Data Engineering & Persistence:**
   - Engineered an automated pipeline to store validation predictions, engine telemetry, and calculated error metrics into a **SQLite Database** (`predictive_maintenance.db`).
   - Executed SQL queries to monitor engine health and isolate highest-error edge cases.

4. **Interactive Analytics Dashboard:**
   - Built an interactive web application using **Streamlit** and **Plotly** to visualize model predictions, equipment degradation lines, and KPI metrics in real-time.

---

##   Tech Stack & Tools

- **Programming Language:** Python
- **Machine Learning:** Scikit-Learn, NumPy, Pandas
- **Data Engineering & SQL:** SQLite, SQLAlchemy
- **Visualization & Frontend:** Streamlit, Plotly
- **Environment & Tools:** Visual Studio Code, Git

---

##  Repository Structure

```text
├── CMaps/                     # NASA CMAPSS Dataset text files
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── Preprocessin.py            # Data loading, RUL calculation & cleaning
├── train_model.py             # Random Forest training & evaluation
├── database.py                # SQL database creation & querying
├── dashboard.py               # Interactive Streamlit application
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation

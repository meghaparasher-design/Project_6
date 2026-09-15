# ============================================================
# STORE SALES FORECASTING
# MODERN INTERACTIVE STREAMLIT DASHBOARD
# ============================================================

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.models import load_model


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DATA_PATH = DATA_DIR / "feature_engineered_store_train.csv"
TEST_DATA_PATH = DATA_DIR / "feature_engineered_store_test.csv"
STORE_METRICS_PATH = DATA_DIR / "per_store_metrics.csv"

RF_MODEL_PATH = MODEL_DIR / "random_forest_sales_model.pkl"

GRU_MODEL_PATH = (
    MODEL_DIR / "multivariate_multistep_gru.keras"
)

FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = MODEL_DIR / "target_scaler.pkl"

LOG_FILE = LOG_DIR / "streamlit_dashboard.log"


# ============================================================
# 2. LOGGING
# ============================================================

logger = logging.getLogger("StoreSales_ModernDashboard")
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(
    LOG_FILE,
    mode="a",
    encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("Modern Store Sales dashboard started.")


# ============================================================
# 3. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="StoreVision AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 4. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */

    .stApp {
        background:
        linear-gradient(
            135deg,
            #f8faff 0%,
            #eef4ff 45%,
            #f7f3ff 100%
        );
    }

    /* Main content width */

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background:
        linear-gradient(
            180deg,
            #111827 0%,
            #172554 50%,
            #312e81 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Hero */

    .hero {
        padding: 34px 38px;
        border-radius: 24px;
        background:
        linear-gradient(
            120deg,
            #2563eb,
            #7c3aed,
            #db2777
        );
        box-shadow:
        0 15px 35px rgba(37, 99, 235, 0.22);
        margin-bottom: 25px;
        color: white;
    }

    .hero h1 {
        color: white;
        font-size: 42px;
        margin: 0;
    }

    .hero p {
        color: #f3f4f6;
        font-size: 17px;
        margin-top: 10px;
        margin-bottom: 0px;
    }

    /* KPI cards */

    .kpi-card {
        background: rgba(255,255,255,0.92);
        border-radius: 20px;
        padding: 22px;
        min-height: 135px;
        box-shadow:
        0px 8px 24px rgba(15,23,42,0.08);
        border: 1px solid rgba(255,255,255,0.7);
        transition: 0.25s;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow:
        0px 14px 32px rgba(37,99,235,0.14);
    }

    .kpi-icon {
        font-size: 28px;
    }

    .kpi-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
        margin-top: 8px;
    }

    .kpi-value {
        color: #0f172a;
        font-size: 27px;
        font-weight: 800;
        margin-top: 4px;
    }

    /* Section heading */

    .section-title {
        font-size: 27px;
        font-weight: 800;
        color: #172554;
        margin-top: 10px;
        margin-bottom: 16px;
    }

    /* Insight box */

    .insight-box {
        padding: 18px 22px;
        background:
        linear-gradient(
            90deg,
            #eff6ff,
            #f5f3ff
        );
        border-left: 5px solid #6366f1;
        border-radius: 14px;
        margin-top: 15px;
        margin-bottom: 20px;
        color: #1e293b;
    }

    /* Model winner */

    .winner-card {
        padding: 24px;
        border-radius: 20px;
        background:
        linear-gradient(
            120deg,
            #059669,
            #10b981
        );
        color: white;
        box-shadow:
        0 10px 30px rgba(16,185,129,0.22);
    }

    .winner-card h2,
    .winner-card h3,
    .winner-card p {
        color: white;
    }

    /* Forecast */

    .forecast-card {
        padding: 22px;
        border-radius: 18px;
        background: white;
        border-top: 5px solid #8b5cf6;
        box-shadow:
        0 8px 24px rgba(15,23,42,0.08);
    }

    /* Footer */

    .footer {
        text-align: center;
        color: #64748b;
        padding-top: 35px;
        padding-bottom: 15px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def kpi_card(icon, title, value):

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title):

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )


def style_figure(fig, height=430):

    fig.update_layout(
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.85)",
        hovermode="x unified",
        font=dict(
            family="Arial",
            size=13
        )
    )

    return fig


# ============================================================
# 6. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    logger.info("Loading feature-engineered training data.")

    df = pd.read_csv(
        TRAIN_DATA_PATH,
        low_memory=False
    )

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

    logger.info("Training data loaded successfully.")

    logger.debug(
        "Training data shape: %s",
        df.shape
    )

    return df


@st.cache_data
def load_store_metrics():

    if STORE_METRICS_PATH.exists():

        return pd.read_csv(
            STORE_METRICS_PATH
        )

    return None


@st.cache_resource
def load_gru_resources():

    logger.info("Loading GRU forecasting resources.")

    model = load_model(
        GRU_MODEL_PATH
    )

    feature_scaler = joblib.load(
        FEATURE_SCALER_PATH
    )

    target_scaler = joblib.load(
        TARGET_SCALER_PATH
    )

    logger.info("GRU resources loaded successfully.")

    return (
        model,
        feature_scaler,
        target_scaler
    )


# ============================================================
# 7. LOAD MAIN DATASET
# ============================================================

try:

    store_train = load_data()

except Exception as error:

    logger.exception(
        "Failed to load training dataset."
    )

    st.error(
        "Training data could not be loaded."
    )

    st.exception(error)

    st.stop()


# ============================================================
# 8. PREPARE DAILY SALES
# ============================================================

if (
    "Date" in store_train.columns
    and "Sales" in store_train.columns
):

    daily_sales = (
        store_train
        .dropna(subset=["Date"])
        .groupby("Date")["Sales"]
        .sum()
        .sort_index()
    )

else:

    daily_sales = pd.Series(dtype=float)


# ============================================================
# 9. SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <h1 style="
        font-size:30px;
        margin-bottom:0px;
    ">
        📊 StoreVision
    </h1>

    <p style="
        opacity:0.75;
        margin-top:3px;
    ">
        AI Sales Intelligence
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Explore Dashboard",
    [
        "🏠 Executive Overview",
        "📈 Sales Analytics",
        "🏪 Store Explorer",
        "🤖 ML Performance",
        "⏱️ Forecasting Models",
        "🔮 7-Day Forecast",
        "📋 Project Summary"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Store Sales Forecasting"
)

st.sidebar.caption(
    "ML • Time Series • Deep Learning"
)


# ============================================================
# 10. EXECUTIVE OVERVIEW
# ============================================================

if page == "🏠 Executive Overview":

    st.markdown(
        """
        <div class="hero">
            <h1>Store Sales Intelligence</h1>
            <p>
                Machine Learning + Time-Series +
                Deep Learning powered sales forecasting
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_sales = store_train["Sales"].sum()

    avg_sales = store_train["Sales"].mean()

    stores = store_train["Store"].nunique()

    if len(daily_sales) > 0:
        avg_daily = daily_sales.mean()
    else:
        avg_daily = 0


    # KPI ROW
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi_card(
            "💰",
            "Total Sales",
            f"{total_sales / 1e9:.2f} B"
        )

    with col2:
        kpi_card(
            "🏪",
            "Stores",
            f"{stores:,}"
        )

    with col3:
        kpi_card(
            "🧾",
            "Average Store-Day Sales",
            f"{avg_sales:,.0f}"
        )

    with col4:
        kpi_card(
            "📅",
            "Average Daily Aggregate Sales",
            f"{avg_daily / 1e6:.2f} M"
        )


    section_title(
        "📈 Daily Sales Journey"
    )

    if len(daily_sales) > 0:

        sales_df = daily_sales.reset_index()

        sales_df.columns = [
            "Date",
            "Sales"
        ]

        fig = px.area(
            sales_df,
            x="Date",
            y="Sales",
            title="Aggregate Sales Over Time",
            template="plotly_white"
        )

        fig.update_traces(
            line=dict(
                width=2
            )
        )

        style_figure(
            fig,
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    col1, col2 = st.columns([1.25, 1])

    with col1:

        section_title(
            "📅 Sales by Day of Week"
        )

        weekday_sales = (
            store_train
            .groupby("DayOfWeek")["Sales"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            weekday_sales,
            x="DayOfWeek",
            y="Sales",
            color="Sales",
            title="Average Sales by Day",
            template="plotly_white",
            color_continuous_scale="Blues"
        )

        style_figure(
            fig,
            height=390
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        section_title(
            "🎯 Promotion Impact"
        )

        promo_sales = (
            store_train
            .groupby("Promo")["Sales"]
            .mean()
            .reset_index()
        )

        promo_sales["Promotion"] = (
            promo_sales["Promo"]
            .map(
                {
                    0: "No Promotion",
                    1: "Promotion"
                }
            )
        )

        fig = px.bar(
            promo_sales,
            x="Promotion",
            y="Sales",
            color="Promotion",
            title="Average Sales: Promo vs No Promo",
            template="plotly_white"
        )

        style_figure(
            fig,
            height=390
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.markdown(
        """
        <div class="insight-box">
        <b>💡 Business Insight:</b>
        Store availability, promotions, competition and
        recurring weekly patterns play important roles in
        predicting sales.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 11. SALES ANALYTICS
# ============================================================

elif page == "📈 Sales Analytics":

    st.title("📈 Interactive Sales Analytics")

    st.write(
        "Explore historical sales patterns using "
        "interactive filters and visualizations."
    )

    if len(daily_sales) == 0:

        st.warning(
            "Daily sales data is unavailable."
        )

    else:

        min_date = daily_sales.index.min().date()
        max_date = daily_sales.index.max().date()

        date_range = st.date_input(
            "📅 Select Date Range",
            value=(
                min_date,
                max_date
            ),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:

            start_date = pd.Timestamp(
                date_range[0]
            )

            end_date = pd.Timestamp(
                date_range[1]
            )

            filtered_sales = daily_sales.loc[
                (daily_sales.index >= start_date)
                &
                (daily_sales.index <= end_date)
            ]

        else:

            filtered_sales = daily_sales


        # KPI ROW
        c1, c2, c3 = st.columns(3)

        with c1:
            kpi_card(
                "💵",
                "Sales in Selected Period",
                f"{filtered_sales.sum() / 1e9:.2f} B"
            )

        with c2:
            kpi_card(
                "📊",
                "Average Daily Sales",
                f"{filtered_sales.mean() / 1e6:.2f} M"
            )

        with c3:
            kpi_card(
                "🚀",
                "Peak Daily Sales",
                f"{filtered_sales.max() / 1e6:.2f} M"
            )


        chart_df = filtered_sales.reset_index()

        chart_df.columns = [
            "Date",
            "Sales"
        ]

        fig = px.line(
            chart_df,
            x="Date",
            y="Sales",
            title="Daily Aggregate Sales",
            template="plotly_white"
        )

        style_figure(
            fig,
            height=480
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # Monthly Sales
        section_title(
            "📆 Monthly Sales Trend"
        )

        monthly = (
            store_train
            .dropna(subset=["Date"])
            .set_index("Date")
            ["Sales"]
            .resample("ME")
            .sum()
            .reset_index()
        )

        fig = px.bar(
            monthly,
            x="Date",
            y="Sales",
            color="Sales",
            color_continuous_scale="Purples",
            title="Monthly Aggregate Sales",
            template="plotly_white"
        )

        style_figure(
            fig,
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# 12. STORE EXPLORER
# ============================================================

elif page == "🏪 Store Explorer":

    #st.title("🏪 Individual Store Explorer")
    st.markdown(
    """
    <h1 style="
        background: linear-gradient(90deg, #2563EB, #7C3AED, #DB2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 10px;
    ">
        🏪 Individual Store Explorer
    </h1>
    """,
    unsafe_allow_html=True
)

    st.write(
        "Select a store to explore its sales history "
        "and performance."
    )

    store_list = sorted(
        store_train["Store"]
        .dropna()
        .unique()
    )

    selected_store = st.selectbox(
        "Select Store",
        store_list
    )

    store_df = (
        store_train[
            store_train["Store"]
            == selected_store
        ]
        .copy()
    )

    if "Date" in store_df.columns:

        store_df = store_df.sort_values(
            "Date"
        )


    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card(
            "🏪",
            "Store",
            str(selected_store)
        )

    with c2:
        kpi_card(
            "💰",
            "Total Sales",
            f"{store_df['Sales'].sum() / 1e6:.2f} M"
        )

    with c3:
        kpi_card(
            "🧾",
            "Average Sales",
            f"{store_df['Sales'].mean():,.0f}"
        )

    with c4:
        kpi_card(
            "🚀",
            "Highest Sales",
            f"{store_df['Sales'].max():,.0f}"
        )


    if "Date" in store_df.columns:

        fig = px.line(
            store_df,
            x="Date",
            y="Sales",
            title=f"Daily Sales - Store {selected_store}",
            template="plotly_white"
        )

        style_figure(
            fig,
            height=470
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    col1, col2 = st.columns(2)

    with col1:

        promo_store = (
            store_df
            .groupby("Promo")["Sales"]
            .mean()
            .reset_index()
        )

        promo_store["Promotion"] = (
            promo_store["Promo"]
            .map(
                {
                    0: "No Promo",
                    1: "Promo"
                }
            )
        )

        fig = px.bar(
            promo_store,
            x="Promotion",
            y="Sales",
            color="Promotion",
            title="Promotion Impact",
            template="plotly_white"
        )

        style_figure(
            fig,
            height=360
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        weekday_store = (
            store_df
            .groupby("DayOfWeek")["Sales"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            weekday_store,
            x="DayOfWeek",
            y="Sales",
            color="Sales",
            color_continuous_scale="Viridis",
            title="Sales by Day of Week",
            template="plotly_white"
        )

        style_figure(
            fig,
            height=360
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# 13. MACHINE LEARNING PERFORMANCE
# ============================================================

elif page == "🤖 ML Performance":

    st.title("🤖 Machine Learning Performance")

    st.markdown(
        """
        <div class="winner-card">
            <h2>🏆 Best Traditional ML Model</h2>
            <h3>Random Forest</h3>
            <p>
            R² = 0.837 &nbsp; | &nbsp;
            MAE = 1,016.85 &nbsp; | &nbsp;
            RMSE = 1,552.79
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")


    ml_results = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "Extra Trees",
                "Linear Regression",
                "Ridge Regression"
            ],
            "R²": [
                0.836961,
                0.827616,
                0.575091,
                0.574931
            ],
            "MAE": [
                1016.8452,
                1022.7639,
                1750.5033,
                1750.9603
            ],
            "RMSE": [
                1552.7875,
                1596.6707,
                2506.7683,
                2507.2403
            ]
        }
    )


    section_title(
        "🏁 Model Leaderboard"
    )

    st.dataframe(
        ml_results.style
        .background_gradient(
            subset=["R²"],
            cmap="Greens"
        )
        .format(
            {
                "R²": "{:.4f}",
                "MAE": "{:,.2f}",
                "RMSE": "{:,.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    fig = px.bar(
        ml_results.sort_values(
            "R²"
        ),
        x="R²",
        y="Model",
        orientation="h",
        color="R²",
        color_continuous_scale="Blues",
        title="Machine Learning Model Comparison",
        template="plotly_white"
    )

    style_figure(
        fig,
        height=410
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Feature Importance
    section_title(
        "🔍 Random Forest Feature Importance"
    )

    importance_df = pd.DataFrame(
        {
            "Feature": [
                "Open",
                "Promo",
                "CompetitionDistance",
                "Store",
                "CompetitionOpenSinceYear",
                "DayOfWeek",
                "CompetitionOpenSinceMonth",
                "Promo2SinceYear",
                "DayOfYear",
                "StoreType_b",
                "Assortment_c",
                "Promo2SinceWeek",
                "CompetitionOpenMonths",
                "Day",
                "StoreType_c"
            ],

            "Importance": [
                0.556913,
                0.088971,
                0.076349,
                0.053487,
                0.029876,
                0.028450,
                0.025341,
                0.020956,
                0.020091,
                0.017441,
                0.015915,
                0.013563,
                0.012839,
                0.007375,
                0.007184
            ]
        }
    )

    importance_df = importance_df.sort_values(
        "Importance"
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Tealgrn",
        title="Top 15 Important Features",
        template="plotly_white"
    )

    style_figure(
        fig,
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.markdown(
        """
        <div class="insight-box">
        <b>💡 Key Insight:</b>
        Whether the store is <b>Open</b> is the strongest
        predictor of sales. Promotions, competition distance
        and store-specific characteristics also contribute
        significantly.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 14. FORECASTING MODEL COMPARISON
# ============================================================

elif page == "⏱️ Forecasting Models":

    st.title("⏱️ Time-Series & Deep Learning")

    st.markdown(
        """
        <div class="winner-card">
            <h2>🏆 Best One-Step Forecasting Model</h2>
            <h3>GRU Neural Network</h3>
            <p>
                Highest R² among the aligned one-step
                forecasting models
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")


    forecast_results = pd.DataFrame(
        {
            "Model": [
                "GRU",
                "SARIMA",
                "LSTM",
                "ARIMA"
            ],

            "R²": [
                0.710235,
                0.610190,
                0.560214,
                0.210973
            ],

            "MAE": [
                1042225,
                1367175,
                1341490,
                2260768
            ],

            "RMSE": [
                1728577,
                2004900,
                2129545,
                2852411
            ]
        }
    )


    c1, c2, c3, c4 = st.columns(4)

    for column, row in zip(
        [c1, c2, c3, c4],
        forecast_results.itertuples()
    ):

        with column:

            kpi_card(
                "🧠",
                row.Model,
                f"R² {row._2:.3f}"
            )


    section_title(
        "📊 R² Score Comparison"
    )

    fig = px.bar(
        forecast_results,
        x="Model",
        y="R²",
        color="R²",
        text="R²",
        color_continuous_scale="Viridis",
        title="Forecasting Model Performance",
        template="plotly_white"
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    style_figure(
        fig,
        height=430
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    section_title(
        "📋 Detailed Metrics"
    )

    st.dataframe(
        forecast_results.style
        .background_gradient(
            subset=["R²"],
            cmap="Greens"
        )
        .format(
            {
                "R²": "{:.4f}",
                "MAE": "{:,.0f}",
                "RMSE": "{:,.0f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        """
        <div class="insight-box">
        <b>💡 Result:</b>
        GRU achieved the strongest one-step forecasting
        performance. SARIMA also performed well because
        it captures the strong 7-day seasonal pattern.
        ARIMA performed weakest because it does not
        explicitly model this weekly seasonality.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 15. 7-DAY FORECAST
# ============================================================

elif page == "🔮 7-Day Forecast":

    st.title("🔮 AI-Powered 7-Day Sales Forecast")

    st.write(
        "Forecast generated using the "
        "**Multivariate Multi-Step GRU model**."
    )


    required_files = [
        GRU_MODEL_PATH,
        FEATURE_SCALER_PATH,
        TARGET_SCALER_PATH
    ]

    missing_files = [
        path.name
        for path in required_files
        if not path.exists()
    ]


    if missing_files:

        st.error(
            "The following forecasting files are missing:"
        )

        for file in missing_files:
            st.write(f"❌ {file}")

        st.info(
            "Run the Deep Learning notebook first "
            "to regenerate the GRU model and scalers."
        )

    else:

        try:

            (
                multi_gru,
                feature_scaler,
                target_scaler
            ) = load_gru_resources()


            # --------------------------------------------
            # Aggregate data
            # --------------------------------------------

            daily_multi = (
                store_train
                .dropna(subset=["Date"])
                .groupby("Date")
                .agg(
                    {
                        "Sales": "sum",
                        "Promo": "mean",
                        "Open": "mean",
                        "SchoolHoliday": "mean",
                        "DayOfWeek": "first"
                    }
                )
                .sort_index()
            )


            feature_columns = [
                "Sales",
                "Promo",
                "Open",
                "SchoolHoliday",
                "DayOfWeek"
            ]


            # --------------------------------------------
            # Last 7 days
            # --------------------------------------------

            input_window = 7

            last_window = daily_multi.tail(
                input_window
            )

            input_features = (
                last_window[
                    feature_columns
                ]
                .values
            )


            # --------------------------------------------
            # Scale
            # --------------------------------------------

            scaled_features = (
                feature_scaler.transform(
                    input_features
                )
            )


            X_future = scaled_features.reshape(
                1,
                input_window,
                len(feature_columns)
            )


            # --------------------------------------------
            # Prediction
            # --------------------------------------------

            prediction_scaled = (
                multi_gru.predict(
                    X_future,
                    verbose=0
                )
                .reshape(-1, 1)
            )


            predictions = (
                target_scaler
                .inverse_transform(
                    prediction_scaled
                )
                .flatten()
            )


            predictions = np.clip(
                predictions,
                0,
                None
            )


            # --------------------------------------------
            # Future dates
            # --------------------------------------------

            last_date = daily_multi.index.max()

            future_dates = pd.date_range(
                start=last_date
                + pd.Timedelta(days=1),
                periods=7,
                freq="D"
            )


            forecast_df = pd.DataFrame(
                {
                    "Date": future_dates,
                    "Predicted Sales": predictions
                }
            )


            # --------------------------------------------
            # KPI cards
            # --------------------------------------------

            c1, c2, c3 = st.columns(3)

            with c1:

                kpi_card(
                    "📊",
                    "7-Day Total Forecast",
                    f"{predictions.sum() / 1e6:.2f} M"
                )

            with c2:

                kpi_card(
                    "📅",
                    "Average Daily Forecast",
                    f"{predictions.mean() / 1e6:.2f} M"
                )

            with c3:

                best_day_index = np.argmax(
                    predictions
                )

                kpi_card(
                    "🚀",
                    "Highest Forecast",
                    f"{predictions[best_day_index] / 1e6:.2f} M"
                )


            # --------------------------------------------
            # Historical + Forecast chart
            # --------------------------------------------

            section_title(
                "📈 Historical Sales + Future Forecast"
            )


            recent_history = (
                daily_multi["Sales"]
                .tail(35)
            )


            fig = go.Figure()


            fig.add_trace(
                go.Scatter(
                    x=recent_history.index,
                    y=recent_history.values,
                    mode="lines",
                    name="Historical Sales",
                    line=dict(
                        width=3
                    )
                )
            )


            fig.add_trace(
                go.Scatter(
                    x=forecast_df["Date"],
                    y=forecast_df[
                        "Predicted Sales"
                    ],
                    mode="lines+markers",
                    name="7-Day Forecast",
                    line=dict(
                        width=4,
                        dash="dash"
                    ),
                    marker=dict(
                        size=9
                    )
                )
            )


            fig.update_layout(
                title=(
                    "Recent Historical Sales "
                    "and 7-Day GRU Forecast"
                ),
                xaxis_title="Date",
                yaxis_title="Aggregate Sales",
                template="plotly_white"
            )


            style_figure(
                fig,
                height=500
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


            # --------------------------------------------
            # Forecast table
            # --------------------------------------------

            section_title(
                "📋 Forecast Details"
            )


            display_forecast = forecast_df.copy()

            display_forecast[
                "Day"
            ] = (
                display_forecast["Date"]
                .dt.day_name()
            )


            display_forecast[
                "Predicted Sales"
            ] = (
                display_forecast[
                    "Predicted Sales"
                ]
                .round()
                .astype(int)
            )


            display_forecast[
                "Date"
            ] = (
                display_forecast[
                    "Date"
                ]
                .dt.strftime(
                    "%d %b %Y"
                )
            )


            display_forecast = (
                display_forecast[
                    [
                        "Date",
                        "Day",
                        "Predicted Sales"
                    ]
                ]
            )


            st.dataframe(
                display_forecast.style.format(
                    {
                        "Predicted Sales":
                        "{:,.0f}"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


            # --------------------------------------------
            # Multivariate model performance
            # --------------------------------------------

            st.markdown(
                """
                <div class="insight-box">
                <b>🧠 Multivariate Multi-Step GRU:</b><br>
                R² = <b>0.7745</b><br>
                MAE ≈ <b>825,256</b><br>
                RMSE ≈ <b>1,528,809</b><br><br>

                This model uses Sales, Promotion,
                Store Open status, School Holiday and
                Day of Week information to generate
                a seven-day sales forecast.
                </div>
                """,
                unsafe_allow_html=True
            )


            logger.info(
                "7-day forecast generated successfully."
            )


        except Exception as error:

            logger.exception(
                "Forecast generation failed."
            )

            st.error(
                "Unable to generate forecast."
            )

            st.exception(error)


# ============================================================
# 16. PROJECT SUMMARY
# ============================================================

elif page == "📋 Project Summary":

    st.title("📋 Project Summary")

    st.markdown(
        """
        <div class="hero">
            <h1>From Data to Decisions</h1>
            <p>
                An end-to-end intelligent Store Sales
                Forecasting solution
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    section_title(
        "🎯 Project Workflow"
    )


    workflow = [
        ("1️⃣", "Data Preprocessing",
         "Cleaned missing values and standardized data"),

        ("2️⃣", "EDA",
         "Analyzed sales trends, stores and promotions"),

        ("3️⃣", "Feature Engineering",
         "Created calendar, competition and promotion features"),

        ("4️⃣", "Machine Learning",
         "Compared RF, Extra Trees, Linear and Ridge"),

        ("5️⃣", "Time-Series",
         "Performed ADF, ACF/PACF, ARIMA and SARIMA"),

        ("6️⃣", "Deep Learning",
         "Developed LSTM and GRU forecasting models"),

        ("7️⃣", "Multi-Step Forecast",
         "Built a multivariate 7-day GRU model"),

        ("8️⃣", "Dashboard",
         "Created an interactive Streamlit application")
    ]


    for icon, title, text in workflow:

        st.markdown(
            f"""
            <div class="kpi-card"
                 style="margin-bottom:12px;
                        min-height:auto;">
                <b style="font-size:18px;">
                    {icon} {title}
                </b>
                <br>
                <span style="color:#64748b;">
                    {text}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )


    section_title(
        "🏆 Final Recommendation"
    )


    st.markdown(
        """
        <div class="winner-card">
            <h2>GRU-Based Forecasting System</h2>

            <p>
            GRU achieved the strongest one-step forecasting
            performance, while the Multivariate Multi-Step
            GRU provides practical seven-day forecasting.
            Combined with the Streamlit dashboard, the
            solution can support data-driven sales planning
            and business decision-making.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 17. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        StoreVision AI • Store Sales Forecasting Dashboard
        <br>
        Machine Learning • Time-Series • Deep Learning
    </div>
    """,
    unsafe_allow_html=True
)
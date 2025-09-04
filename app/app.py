import os

import mlflow
import pandas as pd
import requests
import streamlit as st
from mlflow.pyfunc import PyFuncModel
from requests.auth import HTTPBasicAuth


# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="House Price ML Inference",
    page_icon="🏡",
    layout="wide",
)


# Specify your Unity Catalog model path (update with your actual catalog, schema, model, and alias/version)
MODEL_URI = "models:/mlops_dev.house_prices.house_prices_model_basic@latest-model"

# Trick to ensure DATABRICKS_HOST is set with 'https://' prefix."""
raw_host = os.environ["DATABRICKS_HOST"]
host = raw_host if raw_host.startswith("https://") else f"https://{raw_host}"


def get_token()-> str:
    """
    Retrieves an OAuth access token from the Databricks workspace.

    :return: The access token string.
    """    
    response = requests.post(
        f"{host}/oidc/v1/token",
        auth=HTTPBasicAuth(
            os.environ["DATABRICKS_CLIENT_ID"], 
            os.environ["DATABRICKS_CLIENT_SECRET"]
            ),
        data={
            'grant_type': 'client_credentials',
             'scope': 'all-apis'
             }
    )

    return response.json()["access_token"]

os.environ["DATABRICKS_TOKEN"] = get_token()

# Set MLflow to use Unity Catalog as the registry
mlflow.set_registry_uri("databricks-uc")


@st.cache_resource
def load_uc_model()-> PyFuncModel:
    """
    Loads a PyFunc model from the specified MLflow model URI.

    :return: The loaded MLflow PyFuncModel.
    """    
    return mlflow.pyfunc.load_model(MODEL_URI)


model = load_uc_model()

# --- SIDEBAR ---
with st.sidebar:
     st.image("./house.png", width=300)
     st.title("🏡 House Price Predictor")
     st.markdown("This app predicts house prices using a Databricks Unity Catalog ML model.")
     st.markdown("**Instructions:**\n- Fill in the property details below\n- Click 'Predict' to get the estimated price")

st.title("ML Inference with Unity Catalog Model (Databricks Apps)")

# --- LAYOUT: MAIN INPUTS IN COLUMNS ---
col1, col2, col3 = st.columns(3)

with col1:
    LotFrontage = st.number_input("Lot Frontage", value=55.0, min_value=0.0, step=10.0, help="Linear feet of street connected to property")
    LotArea     = st.number_input("Lot Area", value=8500, min_value=0, step=10, help="Lot size in square feet")
    OverallQual = st.number_input("Overall Quality", value=5, min_value=1, max_value=10, step=1, help="Overall material and finish quality")
    OverallCond = st.number_input("Overall Condition", min_value=1, max_value=10, value=5, help="Overall condition rating")
    YearBuilt   = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000, help="Original construction date")

with col2:
    YearRemodAdd = st.number_input("Year Remodeled", min_value=1800, max_value=2025, value=2000, help="Remodel date (if any)")
    MasVnrArea   = st.number_input("Masonry Veneer Area", min_value=0.0, value=100.0, help="Masonry veneer area in square feet")
    TotalBsmtSF  = st.number_input("Total Basement SF", min_value=0, value=100, help="Total basement area in square feet")
    GrLivArea    = st.number_input("Above Ground Living Area", min_value=0, value=1000, help="Living area above ground in square feet")
    GarageCars   = st.number_input("Garage Cars", min_value=0, max_value=10, value=1, help="Size of garage in car capacity")

with col3:
    MSZoning = st.selectbox("MS Zoning", options=['RL', 'RM', 'FV', 'RH', 'C (all)'], help="General zoning classification")
    Street = st.selectbox("Street Type", options=['Pave', 'Grvl'], help="Type of road access")
    Alley = st.selectbox("Alley Access", options=['NoAlley', 'Grvl', 'Pave'], help="Type of alley access")
    LotShape = st.selectbox("Lot Shape", options=['Reg', 'IR1', 'IR2', 'IR3'], help="General shape of property")
    LandContour = st.selectbox("Land Contour", options=['Lvl', 'Bnk', 'HLS', 'Low'], help="Flatness of the property")

# --- EXPANDER FOR ADVANCED/CATEGORICAL FIELDS ---
with st.expander("Show More Property Details", expanded=False):
    Neighborhood = st.selectbox("Neighborhood", options=['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel'])
    Condition1 = st.selectbox("Proximity to Main Road/Railroad", options=['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA', 'RRNe'])
    BldgType = st.selectbox("Building Type", options=['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'])
    HouseStyle = st.selectbox("House Style", options=['1Story', '2Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'])
    RoofStyle = st.selectbox("Roof Style", options=['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed'])
    Exterior1st = st.selectbox("Exterior Covering 1", options=['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'Stucco', 'Other'])
    Exterior2nd = st.selectbox("Exterior Covering 2", options=['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'Stucco', 'Other'])
    MasVnrType = st.selectbox("Masonry Veneer Type", options=['BrkFace', 'None', 'Stone', 'BrkCmn'])
    Foundation = st.selectbox("Foundation", options=['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'])
    Heating = st.selectbox("Heating", options=['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'])
    CentralAir = st.selectbox("Central Air", options=['Y', 'N'])
    SaleType = st.selectbox("Sale Type", options=['WD', 'New', 'COD', 'ConLD', 'ConLI', 'ConLw', 'Con', 'CWD', 'Oth'])
    SaleCondition = st.selectbox("Sale Condition", options=['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'])

# --- DATAFRAME PREPARATION ---
input_df = pd.DataFrame([[
    LotFrontage, LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, MasVnrArea, TotalBsmtSF, GrLivArea, GarageCars,
    MSZoning, Street, Alley, LotShape, LandContour, Neighborhood, Condition1, BldgType,
    HouseStyle, RoofStyle, Exterior1st, Exterior2nd, MasVnrType, Foundation, Heating,
    CentralAir, SaleType, SaleCondition
]], columns=[
    "LotFrontage", "LotArea", "OverallQual",'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF', 'GrLivArea', 'GarageCars',
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Neighborhood', 'Condition1', 'BldgType',
    'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
    'CentralAir', 'SaleType', 'SaleCondition'
])

# --- PREDICTION BUTTON ---
st.markdown("---")
if st.button("🔮 Predict House Price"):
    prediction = model.predict(input_df)
    st.success(f"🏷️ Prediction: ${int(prediction[0]):,}")


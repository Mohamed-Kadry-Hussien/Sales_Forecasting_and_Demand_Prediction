import streamlit as st
import requests

st.set_page_config(page_title="Sales Forecasting and Demand Prediction", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #FAF9EE,white);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    background: transparent !important;

div.stMarkdown h4 ,div.stMarkdown h2 {
        color: #5D866C !important;
            font-weight: bold !important;
        }
            
.stButton button {
        background: linear-gradient(135deg, #5D866C 0%, #B6CEB4 100%);
        color: white;
        font-size: 35px !important;
        font-weight: bold !important;
        padding: 15px 60px !important;
        border-radius: 60px !important;
        border: none;
        width: 100%;
        height: 80px;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
        background: linear-gradient(135deg, #B6CEB4 0%, #5D866C 100%);
    }
}
    
div.stAlert.stAlertSuccess {
    background-color: #e7f5ef !important; 
    border-left: 6px solid #0e503a !important;
    color: #0e503a !important;          
    border-radius: 8px !important;
    padding: 12px !important;
    font-size: 16px !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div style="background-color:#5D866C; padding:10px; border-radius:10px">
<h1 style='color: white; text-align:center;'>Sales Forecasting and Demand Prediction</h1>

</div>
""", unsafe_allow_html=True)


st.markdown("<h2 style='color:white; margin-top:20px;'>Input Features</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("<h4 style='color:white;'>General Inputs</h4>", unsafe_allow_html=True)
    date = st.date_input("Order Date")
    discount = st.number_input("Discount", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    shipping_cost = st.number_input("Shipping Cost", min_value=0.0, value=5.0, step=0.1)
    seasons = ["Autumn", "Spring", "Summer", "Winter"]
    selected_season = st.selectbox("Season", seasons)

with col2:
    st.markdown("<h4 style='color:white;'>Product Details</h4>", unsafe_allow_html=True)
    categories = ["Furniture", "Office Supplies", "Technology"]
    selected_category = st.selectbox("Category", categories)
    sub_categories = ["Accessories", "Appliances", "Art", "Binders", "Bookcases", "Chairs",
                      "Copiers", "Envelopes", "Fasteners", "Furnishings", "Labels", "Machines",
                      "Paper", "Phones", "Storage", "Supplies", "Tables"]
    selected_sub_category = st.selectbox("Sub-Category", sub_categories)
    ship_modes = ["First Class", "Same Day", "Second Class", "Standard Class"]
    selected_ship_mode = st.selectbox("Ship Mode", ship_modes)

with col3:
    st.markdown("<h4 style='color:white;'>Customer & Location</h4>", unsafe_allow_html=True)
    markets = ["APAC", "Africa", "Canada", "EMEA", "EU", "LATAM", "US"]
    selected_market = st.selectbox("Market", markets)
    regions = ["Africa", "Canada", "Caribbean", "Central America", "Central Asia", "EMEA",
               "Eastern US", "North America", "North Asia", "Oceania", "South America",
               "Southeast Asia", "Western US"]
    selected_region = st.selectbox("Region", regions)
    segments = ["Consumer", "Corporate", "Home Office"]
    selected_segment = st.selectbox("Segment", segments)

payload = {
    "order_date": str(date),
    "discount": discount,
    "category": selected_category,
    "sub_category": selected_sub_category,
    "ship_mode": selected_ship_mode,
    "market": selected_market,
    "region": selected_region,
    "segment": selected_segment,
    "season": selected_season
}

st.markdown("<br><br>", unsafe_allow_html=True)
center = st.columns([1, 0.5, 1])
with center[1]:
    if st.button("make Prediction"):
        with st.spinner('Predicting sales and Quantity, please wait... ⏳'):
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Predicted Sales: {data['predicted_sales']:.2f}")
                    st.success(f"Predicted Quantity: {data['predicted_quantity']:.2f}")
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

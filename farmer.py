import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------- Page Config ----------------
st.set_page_config(page_title="Farmer MSP Dashboard", page_icon="🌾", layout="wide")

# ---------------- Custom CSS ----------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f9fff4, #e8ffe8);
}
[data-testid="stHeader"] {background-color: rgba(0,0,0,0);}
.big-title {
    text-align: center;
    font-size: 46px;
    font-weight: bold;
    color: #2E7D32;
    margin-bottom: 10px;
}
.card {
    background-color: white;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    text-align: center;
}
.card h2 {
    color: #2E7D32;
    font-size: 28px;
}
.card p {
    font-size: 20px;
    font-weight: bold;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown('<div class="big-title">🌾 Farmer MSP Analytics Dashboard 🌾</div>', unsafe_allow_html=True)
st.write("This dashboard helps farmers **understand Minimum Support Price (MSP)**, "
         "see price history, predict future trends, and explore data in **3D**.")

# ---------------- Load Data ----------------
df = pd.read_csv("MSP_extracted.csv")
df.columns = df.columns.str.replace("\n", " ")

df_melt = df.melt(id_vars=["Commodities"], var_name="Year", value_name="MSP")
df_melt["Year"] = df_melt["Year"].str.replace("KMS ", "").str.strip()
df_melt = df_melt[df_melt["Year"].str[:4].str.isdigit()]
df_melt["MSP"] = pd.to_numeric(df_melt["MSP"], errors="coerce")
df_melt["Year_num"] = df_melt["Year"].str[:4].astype(int)

# ---------------- Sidebar Filters ----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2913/2913461.png", width=100)
st.sidebar.header("🔎 Filters")
selected_crop = st.sidebar.selectbox("Choose Crop", df_melt["Commodities"].dropna().unique())
year_range = st.sidebar.slider("Select Year Range",
                               int(df_melt["Year_num"].min()),
                               int(df_melt["Year_num"].max()),
                               (int(df_melt["Year_num"].min()), int(df_melt["Year_num"].max())))

crop_df = df_melt[df_melt["Commodities"] == selected_crop].dropna()
crop_df = crop_df[(crop_df["Year_num"] >= year_range[0]) & (crop_df["Year_num"] <= year_range[1])]

# ---------------- Main Choice ----------------
choice = st.radio("📌 Select View", ["📊 Overview", "🔮 Future Prediction", "🌐 3D Visualization"])

# ---------------- Overview ----------------
if choice == "📊 Overview":
    st.subheader(f"🌱 Overview for {selected_crop}")

    if not crop_df.empty:
        latest_price = crop_df.iloc[-1]["MSP"]
        first_price = crop_df.iloc[0]["MSP"]
        avg_price = crop_df["MSP"].mean()
        growth = ((latest_price - first_price) / first_price) * 100 if first_price > 0 else 0
    else:
        latest_price = first_price = avg_price = growth = 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='card'><h2>🌱 Latest MSP</h2><p>₹{latest_price:,.0f}</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='card'><h2>📅 First MSP</h2><p>₹{first_price:,.0f}</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='card'><h2>📊 Average MSP</h2><p>₹{avg_price:,.0f}</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='card'><h2>📈 Growth %</h2><p>{growth:.2f}%</p></div>", unsafe_allow_html=True)

    # Trend Line Chart
    st.subheader(f"📈 MSP Trends for {selected_crop}")
    fig_line = px.line(crop_df, x="Year", y="MSP", markers=True,
                       labels={"MSP": "MSP (₹)", "Year": "Year"},
                       title=f"MSP Trend: {selected_crop}")
    st.plotly_chart(fig_line, use_container_width=True)

    # Bar Chart
    fig_bar = px.bar(crop_df, x="Year", y="MSP", text="MSP", color="MSP",
                     labels={"MSP": "MSP (₹)", "Year": "Year"},
                     title="Year-wise MSP (Bar Chart)")
    fig_bar.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------- Future Prediction ----------------
elif choice == "🔮 Future Prediction":
    st.subheader(f"🔮 MSP Prediction for {selected_crop}")

    if len(crop_df) > 2:
        X = crop_df[["Year_num"]]
        y = crop_df["MSP"]
        model = LinearRegression().fit(X, y)

        next_year = X["Year_num"].max() + 1
        predicted_price = model.predict([[next_year]])[0]

        st.success(f"👉 Expected MSP for {next_year}-{str(next_year+1)[-2:]}: ₹{predicted_price:.0f}")

        X_range = np.linspace(X.min(), next_year, 100).reshape(-1, 1)
        y_pred_line = model.predict(X_range)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=crop_df["Year_num"], y=crop_df["MSP"],
                                      mode="markers+lines", name="Actual MSP"))
        fig_pred.add_trace(go.Scatter(x=X_range.flatten(), y=y_pred_line,
                                      mode="lines", name="Trend Line"))
        fig_pred.add_trace(go.Scatter(x=[next_year], y=[predicted_price],
                                      mode="markers+text",
                                      marker=dict(size=12, color="red"),
                                      text=[f"Predicted {predicted_price:.0f}"],
                                      textposition="top center"))
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.warning("Not enough data for prediction.")

# ---------------- 3D Visualization ----------------
elif choice == "🌐 3D Visualization":
    st.subheader("🌐 3D View of MSP Across Crops and Years")
    all_crops = df_melt.dropna()
    fig_3d = px.scatter_3d(all_crops, x="Year_num", y="Commodities", z="MSP",
                           color="MSP", size="MSP",
                           labels={"Year_num": "Year", "MSP": "MSP (₹)", "Commodities": "Crop"},
                           title="3D MSP Landscape")
    st.plotly_chart(fig_3d, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("""
---
<p style='text-align: center; color: gray;'>
Made with ❤️ for Farmers • Simple Visuals • Clear Insights 🌾
</p>
""", unsafe_allow_html=True)

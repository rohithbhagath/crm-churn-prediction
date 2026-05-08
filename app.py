# ============================================
#   CRM Churn Prediction Dashboard
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="CRM Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Load Assets ──────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('segmented_churn_data.csv')

@st.cache_resource
def load_models():
    model  = joblib.load('best_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

df            = load_data()
model, scaler = load_models()

# ── Encode dataframe for display stats ───────
df_encoded = df.copy()
cat_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c not in ['RiskSegment']]
le = LabelEncoder()
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# ── Sidebar ───────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=80)
st.sidebar.title("CRM Churn System")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🔍 Churn Analysis",
    "🧩 Risk Segments",
    "🔮 Predict Churn"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Info**")
st.sidebar.metric("Total Customers", len(df))
st.sidebar.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")


# ============================================
#   HELPER: Tenure Group
# ============================================
def get_tenure_group(t):
    if t <= 12:
        return 0
    elif t <= 24:
        return 1
    elif t <= 48:
        return 2
    else:
        return 3


# ============================================
#   PAGE 1: OVERVIEW
# ============================================
if page == "📊 Overview":

    st.title("📊 CRM Churn Analytics Dashboard")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    total      = len(df)
    churned    = df['Churn'].sum()
    retained   = total - churned
    churn_rate = df['Churn'].mean() * 100
    avg_clv    = df['CLV'].mean()

    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Churned",         f"{churned:,}",  delta=f"-{churn_rate:.1f}%", delta_color="inverse")
    col3.metric("Retained",        f"{retained:,}", delta=f"+{100-churn_rate:.1f}%")
    col4.metric("Avg CLV",         f"${avg_clv:,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        churn_counts = df['Churn'].value_counts().reset_index()
        churn_counts.columns = ['Churn', 'Count']
        churn_counts['Churn'] = churn_counts['Churn'].map({0: 'Retained', 1: 'Churned'})
        fig = px.pie(
            churn_counts, values='Count', names='Churn',
            title='Churn Distribution',
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df, x='MonthlyCharges', color='Churn',
            barmode='overlay',
            title='Monthly Charges Distribution',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            df, x='RiskSegment', y='CLV',
            color='RiskSegment',
            title='CLV by Risk Segment',
            color_discrete_map={
                'High Risk'  : '#e74c3c',
                'Medium Risk': '#f39c12',
                'Low Risk'   : '#2ecc71'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df, x='tenure', color='Churn',
            barmode='overlay',
            title='Tenure Distribution by Churn',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================
#   PAGE 2: CHURN ANALYSIS
# ============================================
elif page == "🔍 Churn Analysis":

    st.title("🔍 Churn Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        contract_churn = df.groupby('Contract')['Churn'].mean().reset_index()
        contract_churn['Churn'] = contract_churn['Churn'] * 100
        fig = px.bar(
            contract_churn, x='Contract', y='Churn',
            title='Churn Rate by Contract Type (%)',
            color='Churn',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        internet_churn = df.groupby('InternetService')['Churn'].mean().reset_index()
        internet_churn['Churn'] = internet_churn['Churn'] * 100
        fig = px.bar(
            internet_churn, x='InternetService', y='Churn',
            title='Churn Rate by Internet Service (%)',
            color='Churn',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            df, x='tenure', y='MonthlyCharges',
            color=df['Churn'].map({0: 'Retained', 1: 'Churned'}),
            title='Tenure vs Monthly Charges',
            color_discrete_map={
                'Retained': '#2ecc71',
                'Churned' : '#e74c3c'
            },
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df,
            x=df['Churn'].map({0: 'Retained', 1: 'Churned'}),
            y='CLV',
            color=df['Churn'].map({0: 'Retained', 1: 'Churned'}),
            title='CLV vs Churn Status',
            color_discrete_map={
                'Retained': '#2ecc71',
                'Churned' : '#e74c3c'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🚨 High Risk Customers")

    display_cols = [c for c in [
        'tenure', 'MonthlyCharges', 'CLV',
        'Contract', 'RiskSegment', 'Churn'
    ] if c in df.columns]

    high_risk = df[df['RiskSegment'] == 'High Risk'][display_cols].head(20)
    st.dataframe(high_risk, use_container_width=True)


# ============================================
#   PAGE 3: RISK SEGMENTS
# ============================================
elif page == "🧩 Risk Segments":

    st.title("🧩 Customer Risk Segmentation")
    st.markdown("---")

    summary = df.groupby('RiskSegment').agg(
        Total      =('Churn', 'count'),
        Churned    =('Churn', 'sum'),
        Churn_Rate =('Churn', lambda x: f"{round(x.mean()*100, 1)}%"),
        Avg_CLV    =('CLV',   lambda x: f"${round(x.mean(), 0):,.0f}"),
        Avg_Tenure =('tenure', lambda x: f"{round(x.mean(), 0)} months")
    )

    if all(s in summary.index for s in ['High Risk', 'Medium Risk', 'Low Risk']):
        summary = summary.reindex(['High Risk', 'Medium Risk', 'Low Risk'])

    st.dataframe(summary, use_container_width=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        seg_counts = df['RiskSegment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        fig = px.pie(
            seg_counts, values='Count', names='Segment',
            title='Customer Distribution by Risk Segment',
            color='Segment',
            color_discrete_map={
                'High Risk'  : '#e74c3c',
                'Medium Risk': '#f39c12',
                'Low Risk'   : '#2ecc71'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_churn = df.groupby('RiskSegment')['Churn'].mean().reset_index()
        seg_churn['Churn'] = seg_churn['Churn'] * 100
        fig = px.bar(
            seg_churn, x='RiskSegment', y='Churn',
            title='Churn Rate by Risk Segment (%)',
            color='RiskSegment',
            color_discrete_map={
                'High Risk'  : '#e74c3c',
                'Medium Risk': '#f39c12',
                'Low Risk'   : '#2ecc71'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 Retention Strategies")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.error("🔴 High Risk")
        st.markdown("""
- Immediate personal outreach
- Offer loyalty discounts
- Dedicated account manager
- Priority complaint resolution
        """)

    with col2:
        st.warning("🟡 Medium Risk")
        st.markdown("""
- Re-engagement email campaigns
- Contract upgrade incentives
- Usage tips & tutorials
- Activity monitoring alerts
        """)

    with col3:
        st.success("🟢 Low Risk")
        st.markdown("""
- Regular satisfaction surveys
- Referral bonus rewards
- Upsell premium features
- Consistent communication
        """)


# ============================================
#   PAGE 4: PREDICT CHURN
# ============================================
elif page == "🔮 Predict Churn":

    st.title("🔮 Predict Customer Churn")
    st.markdown("---")
    st.markdown("Fill in customer details below to predict churn probability.")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_charges   = st.number_input("Total Charges ($)", 0.0, 9000.0, 800.0)
        senior_citizen  = st.selectbox("Senior Citizen", [0, 1])

    with col2:
        contract = st.selectbox("Contract", [
            'Month-to-month', 'One year', 'Two year'
        ])
        internet = st.selectbox("Internet Service", [
            'DSL', 'Fiber optic', 'No'
        ])
        payment = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)',
            'Credit card (automatic)'
        ])
        num_services = st.slider("Number of Services", 0, 9, 3)

    with col3:
        partner      = st.selectbox("Partner",           ['Yes', 'No'])
        dependents   = st.selectbox("Dependents",        ['Yes', 'No'])
        paperless    = st.selectbox("Paperless Billing", ['Yes', 'No'])
        tech_support = st.selectbox("Tech Support",      ['Yes', 'No', 'No internet service'])

    st.markdown("---")

    if st.button("🔮 Predict Churn", use_container_width=True):

        clv               = monthly_charges * tenure
        avg_monthly_spend = total_charges / tenure if tenure > 0 else 0
        contract_risk     = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}[contract]
        elec_check_flag   = 1 if payment == 'Electronic check' else 0
        high_spend_flag   = 1 if monthly_charges > df['MonthlyCharges'].mean() else 0

        internet_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
        partner_map  = {'Yes': 1, 'No': 0}
        dep_map      = {'Yes': 1, 'No': 0}
        paper_map    = {'Yes': 1, 'No': 0}
        tech_map     = {'Yes': 2, 'No': 1, 'No internet service': 0}

        input_data = pd.DataFrame([{
            'tenure'             : tenure,
            'MonthlyCharges'     : monthly_charges,
            'TotalCharges'       : total_charges,
            'CLV'                : clv,
            'AvgMonthlySpend'    : avg_monthly_spend,
            'ServiceCount'       : num_services,
            'ContractRiskScore'  : contract_risk,
            'ElectronicCheckFlag': elec_check_flag,
            'HighSpendFlag'      : high_spend_flag,
            'SeniorCitizen'      : senior_citizen,
            'Partner'            : partner_map[partner],
            'Dependents'         : dep_map[dependents],
            'PaperlessBilling'   : paper_map[paperless],
            'TechSupport'        : tech_map[tech_support],
            'InternetService'    : internet_map[internet],
            'TenureGroup'        : get_tenure_group(tenure)
        }])

        if hasattr(scaler, 'feature_names_in_'):
            scaler_features = scaler.feature_names_in_.tolist()
        else:
            drop_cols = ['Churn', 'RiskSegment', 'Cluster']
            drop_cols = [c for c in drop_cols if c in df_encoded.columns]
            scaler_features = df_encoded.drop(columns=drop_cols).columns.tolist()

        for col in scaler_features:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data   = input_data[scaler_features]
        input_scaled = scaler.transform(input_data)
        churn_prob   = model.predict_proba(input_scaled)[0][1]
        churn_pred   = model.predict(input_scaled)[0]

        if churn_prob >= 0.7:
            risk = "🔴 High Risk"
        elif churn_prob >= 0.4:
            risk = "🟡 Medium Risk"
        else:
            risk = "🟢 Low Risk"

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Churn Probability", f"{churn_prob*100:.1f}%")
        col2.metric("Prediction",        "Will Churn" if churn_pred == 1 else "Will Stay")
        col3.metric("Risk Level",        risk)

        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = churn_prob * 100,
            title = {'text': "Churn Probability (%)"},
            gauge = {
                'axis' : {'range': [0, 100]},
                'bar'  : {'color': "darkred"},
                'steps': [
                    {'range': [0,  40],  'color': '#2ecc71'},
                    {'range': [40, 70],  'color': '#f39c12'},
                    {'range': [70, 100], 'color': '#e74c3c'}
                ],
                'threshold': {
                    'line'     : {'color': 'black', 'width': 4},
                    'thickness': 0.75,
                    'value'    : churn_prob * 100
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Recommended Action")

        if churn_prob >= 0.7:
            st.error("""
**Immediate Action Required:**
- Assign dedicated account manager
- Offer loyalty discount or plan upgrade
- Resolve any open complaints immediately
- Schedule personal follow-up call
            """)
        elif churn_prob >= 0.4:
            st.warning("""
**Proactive Engagement Needed:**
- Send re-engagement email campaign
- Offer contract upgrade incentive
- Share product tips and tutorials
- Monitor activity closely
            """)
        else:
            st.success("""
**Maintain Relationship:**
- Send satisfaction survey
- Offer referral bonus program
- Upsell premium features
- Keep regular communication
            """)
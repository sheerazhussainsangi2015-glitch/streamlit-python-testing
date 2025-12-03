import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("📡 Device Online/Offline Downtime Report")

# ------------------------- FILE UPLOAD -------------------------------
uploaded_file = st.file_uploader("📤 Upload Device Log CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# ------------------------- LOAD DATA ---------------------------------
df = pd.read_csv(uploaded_file)

# Your main logic starts here
df['Record Time Format'] = pd.to_datetime(
    df['Record Time'], 
    dayfirst=True,
    errors='coerce'
)
df['Record Time'] = df['Record Time Format']
del(df['Record Time Format'])
df = df[df['Type'].str.contains('encoding', case=False, na=False)]

df['status'] = 'unknown'
df.loc[df['Type'].str.contains('online', case=False, na=False), 'status'] = 'online'
df.loc[df['Type'].str.contains('offline', case=False, na=False), 'status'] = 'offline'
del(df['Type'])

df = df.sort_values(by=['Device Name', 'Record Time'], ascending=[True, True])

# ---------------------- Helper Function ----------------------
def format_duration(seconds):
    if pd.isna(seconds):
        return ""
    seconds = int(seconds)
    d = seconds // 86400
    h = (seconds % 86400) // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{d}d {h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"

# ---------------------- Downtime Calculations ----------------------
df_downtime = (
    df.assign(
        next_status=df.groupby('Device Name')['status'].shift(-1),
        next_time=df.groupby('Device Name')['Record Time'].shift(-1),
        prev_status=df.groupby('Device Name')['status'].shift(1)
    )
    .loc[lambda x: x['status'] == 'offline']
    .assign(
        Downtime_Seconds=lambda x: np.where(
            x['next_status'] == 'online',
            (x['next_time'] - x['Record Time']).dt.total_seconds(),
            np.nan
        ),
        Downtime_Status=lambda x: np.where(
            x['next_status'] == 'online',
            'Completed',
            np.where(x['prev_status'] == 'online', 'Ongoing', 'Intermediate')
        )
    )
    .rename(columns={
        'Record Time': 'Offline_Time',
        'next_time': 'Online_Time',
        'Device Name': 'Device'
    })
    [['Device', 'Offline_Time', 'Online_Time', 'Downtime_Seconds', 'Downtime_Status']]
)

mask = (df_downtime['Online_Time'].notna()) & (df_downtime['Downtime_Status'] == 'Ongoing')
df_downtime.loc[mask, 'Downtime_Status'] = 'Completed'

current_time = pd.Timestamp.now()

def recalc(row):
    if row['Downtime_Status'] == 'Completed':
        return row['Downtime_Seconds']
    elif pd.notna(row['Online_Time']):
        return (row['Online_Time'] - row['Offline_Time']).total_seconds()
    else:
        return (current_time - row['Offline_Time']).total_seconds()

df_downtime['Downtime_Seconds'] = df_downtime.apply(recalc, axis=1)
df_downtime['Downtime_Duration'] = df_downtime['Downtime_Seconds'].apply(format_duration)
df_downtime['Downtime_Status'] = np.where(df_downtime['Downtime_Status'] == 'Ongoing', '🔴 Ongoing', '✔️ Completed')

# ---------------------- Summary Table ----------------------
analysis_time = pd.Timestamp.now()

summary = (
    df_downtime.groupby('Device')
    .agg({
        'Offline_Time': 'last',
        'Online_Time': 'last',
        'Downtime_Seconds': ['count', 'sum'],
        'Downtime_Status': lambda x: (x == '🔴 Ongoing').sum()
    })
    .reset_index()
)

summary.columns = [
    'Device', 'Last_Offline_Time', 'Last_Online_Time',
    'Total_DownTime_Events', 'Total_Downtime_Seconds', 'Ongoing_Count'
]

summary['Current_Downtime_Seconds'] = np.where(
    summary['Ongoing_Count'] > 0,
    (analysis_time - summary['Last_Offline_Time']).dt.total_seconds(),
    np.nan
)

ongoing_mask = summary['Ongoing_Count'] > 0
summary.loc[ongoing_mask, 'Total_Downtime_Seconds'] = np.maximum(
    summary.loc[ongoing_mask, 'Total_Downtime_Seconds'],
    summary.loc[ongoing_mask, 'Current_Downtime_Seconds']
)

summary['Current_Downtime_Duration'] = summary['Current_Downtime_Seconds'].apply(format_duration)
summary['Total_Downtime_Duration'] = summary['Total_Downtime_Seconds'].apply(format_duration)

summary['Current_Status'] = np.where(summary['Ongoing_Count'] > 0, '🔴 Offline', '✔️ Online')

summary = summary[[
    'Device', 'Current_Status', 'Ongoing_Count',
    'Last_Offline_Time', 'Total_DownTime_Events',
    'Current_Downtime_Duration', 'Total_Downtime_Duration'
]]

# ---------------------- INFO CARDS ----------------------
total_encoding = summary['Device'].nunique()
total_devices = df['Device Name'].nunique()
offline_now = (summary['Current_Status'] == '🔴 Offline').sum()
completed_events = (df_downtime['Downtime_Status'] == '✔️ Completed').sum()

colA, colB, colC, colD = st.columns(4)

colA.metric("📟 Encoding Devices", total_encoding)
colB.metric("🎥 Device count csv", total_devices)
colC.metric("🔴 Offline Now", offline_now)
colD.metric("✔️ Completed Events", completed_events)

# ---------------------- Device Filter UI ----------------------
st.subheader("🔍 Device Filter")

device_list = sorted(summary["Device"].unique())

# Initialize session state with validation
if "selected_devices" not in st.session_state:
    st.session_state.selected_devices = device_list
else:
    # Filter out any previously selected devices that don't exist in current device_list
    valid_devices = [device for device in st.session_state.selected_devices if device in device_list]
    # If no valid devices remain, default to all devices
    if not valid_devices and device_list:
        st.session_state.selected_devices = device_list
    else:
        st.session_state.selected_devices = valid_devices

col1, col2 = st.columns([4,1])

with col1:
    selected = st.multiselect(
        "Select Device(s):",
        device_list,
        default=st.session_state.selected_devices
    )

with col2:
    if st.button("Clear Filters"):
        st.session_state.selected_devices = device_list
        st.rerun()

# Update session state with current selection
st.session_state.selected_devices = selected

# Filter data based on selection
if selected:
    df_downtime_filtered = df_downtime[df_downtime["Device"].isin(selected)]
    summary_filtered = summary[summary["Device"].isin(selected)]
else:
    # If nothing selected, show all devices
    df_downtime_filtered = df_downtime
    summary_filtered = summary

# ---------------------- TABS ----------------------
tab1, tab2 = st.tabs(["📄 Downtime Records", "📊 Summary"])

with tab1:
    st.subheader("📄 Detailed Downtime Log")
    st.dataframe(df_downtime_filtered, use_container_width=True, height=700)

with tab2:
    st.subheader("📊 Device Downtime Summary")
    st.dataframe(summary_filtered, use_container_width=True, height=700)

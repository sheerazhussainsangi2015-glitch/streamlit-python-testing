import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import time
from streamlit.components.v1 import html

# Page configuration
st.set_page_config(
    page_title="Device Downtime Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .active-devices {
        color: #4CAF50 !important;
    }
    .live-clock {
        background-color: #f0f8ff;
        padding: 10px 20px;
        border-radius: 8px;
        border: 2px solid #1E88E5;
        font-family: monospace;
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .filter-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    .download-btn {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 5px !important;
        font-weight: bold !important;
    }
    .stats-badge {
        background-color: #e3f2fd;
        color: #1E88E5;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Format duration function
def format_duration(seconds):
    if pd.isna(seconds) or seconds == 0:
        return ""
    seconds = int(seconds)
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Function to process CSV data
@st.cache_data
def process_csv(file, date_range_start=None, date_range_end=None):
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Store original stats
        original_stats = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'total_devices': df['Device Name'].nunique() if 'Device Name' in df.columns else 0
        }
        
        # Process Record Time
        df['Record Time Format'] = pd.to_datetime(
            df['Record Time'], 
            dayfirst=True,
            errors='coerce'
        )
        df['Record Time'] = df['Record Time Format']
        del df['Record Time Format']
        
        # Filter only encoding types
        df = df[df['Type'].str.contains('encoding', case=False, na=False)]
        
        # Create status column
        df['status'] = 'unknown'
        df.loc[df['Type'].str.contains('online', case=False, na=False), 'status'] = 'online'
        df.loc[df['Type'].str.contains('offline', case=False, na=False), 'status'] = 'offline'
        del df['Type']
        
        # Sort by device and time
        df = df.sort_values(by=['Device Name', 'Record Time'], ascending=[True, True])
        
        # Apply date range filter if specified
        if date_range_start and date_range_end:
            mask = (df['Record Time'] >= date_range_start) & (df['Record Time'] <= date_range_end)
            df = df.loc[mask]
        
        # Get processed devices count
        processed_devices = df['Device Name'].nunique() if not df.empty else 0
        
        return df, original_stats, processed_devices
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None, 0

# Function to calculate downtime
@st.cache_data
def calculate_downtime(df):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        # Calculate downtime
        df_downtime = (
            df
            .assign(next_status=df.groupby('Device Name')['status'].shift(-1),
                    next_time=df.groupby('Device Name')['Record Time'].shift(-1),
                    prev_status=df.groupby('Device Name')['status'].shift(1))
            
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
                    np.where(
                        x['prev_status'] == 'online',
                        'Ongoing',
                        'Intermediate'
                    )
                )
            )
            .rename(columns={'Record Time': 'Offline_Time', 'next_time': 'Online_Time',
                           'Device Name': 'Device'})
            [['Device', 'Offline_Time', 'Online_Time', 'Downtime_Seconds', 'Downtime_Status']]
        )
        
        # Fix misclassified records
        mask = (df_downtime['Online_Time'].notna()) & (df_downtime['Downtime_Status'] == 'Ongoing')
        df_downtime.loc[mask, 'Downtime_Status'] = 'Completed'
        
        # Recalculate Downtime_Seconds
        current_time = pd.Timestamp.now()
        
        def recalculate_downtime(row):
            if row['Downtime_Status'] == 'Completed':
                return row['Downtime_Seconds']
            elif pd.notna(row['Online_Time']):
                return (row['Online_Time'] - row['Offline_Time']).total_seconds()
            else:
                return (current_time - row['Offline_Time']).total_seconds()
        
        df_downtime['Downtime_Seconds'] = df_downtime.apply(recalculate_downtime, axis=1)
        df_downtime['Downtime_Duration'] = df_downtime['Downtime_Seconds'].apply(format_duration)
        
        # Create summary
        analysis_time = pd.Timestamp.now()
        
        summary = (
            df_downtime.groupby('Device')
            .agg({
                'Offline_Time': 'last',
                'Online_Time': 'last',
                'Downtime_Seconds': ['count', 'sum'],
                'Downtime_Status': lambda x: (x == 'Ongoing').sum()
            })
            .reset_index()
        )
        
        summary.columns = [
            'Device', 'Last_Offline_Time', 'Last_Online_Time',
            'Total_DownTime_Events', 'Total_Downtime_Seconds', 'Ongoing_Count'
        ]
        
        summary['Total_Downtime_Seconds'] = summary['Total_Downtime_Seconds'].round(0)
        summary['Current_Downtime_Seconds'] = np.where(
            summary['Ongoing_Count'] > 0,
            (analysis_time - summary['Last_Offline_Time']).dt.total_seconds().round(0),
            np.nan
        )
        
        # Ensure Total >= Current for ongoing devices
        ongoing_mask = summary['Ongoing_Count'] > 0
        summary.loc[ongoing_mask, 'Total_Downtime_Seconds'] = np.maximum(
            summary.loc[ongoing_mask, 'Total_Downtime_Seconds'],
            summary.loc[ongoing_mask, 'Current_Downtime_Seconds']
        )
        
        # Format durations
        summary['Current_Downtime_Duration'] = summary['Current_Downtime_Seconds'].apply(format_duration)
        summary['Total_Downtime_Duration'] = summary['Total_Downtime_Seconds'].apply(format_duration)
        
        # Add current status column
        summary['Current_Status'] = np.where(summary['Ongoing_Count'] > 0, '🔴 Offline', '✔️ Online')
        
        # Select final columns
        summary = summary[['Device', 'Current_Status', 'Ongoing_Count', 'Last_Offline_Time', 
                          'Total_DownTime_Events', 'Current_Downtime_Duration', 'Total_Downtime_Duration']]
        
        # Format Downtime_Status for display
        df_downtime['Downtime_Status'] = np.where(df_downtime['Downtime_Status'] == 'Ongoing', 
                                                  '🔴 Ongoing', '✔️ Completed')
        
        return summary, df_downtime
        
    except Exception as e:
        st.error(f"Error calculating downtime: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Function to convert dataframe to Excel
def to_excel(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in df_dict.items():
            if not df.empty:
                df.to_excel(writer, index=False, sheet_name=sheet_name)
    processed_data = output.getvalue()
    return processed_data

# Function to get month ranges
def get_month_range(period, year=None, month=None):
    now = datetime.now()
    if period == "This Month":
        start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = now
    elif period == "Last Month":
        if now.month == 1:
            start_date = now.replace(year=now.year-1, month=12, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            start_date = now.replace(month=now.month-1, day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Get last day of last month
        if start_date.month == 12:
            next_month = start_date.replace(year=start_date.year+1, month=1, day=1)
        else:
            next_month = start_date.replace(month=start_date.month+1, day=1)
        end_date = next_month - timedelta(days=1)
        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    return start_date, end_date

# Header with live clock
col1, col2, col3 = st.columns([3, 1, 2])
with col1:
    st.markdown('<h1 class="main-header">📊 Device Downtime Analyzer</h1>', unsafe_allow_html=True)
with col3:
    # JavaScript for live clock
    clock_js = """
    <script>
    function updateClock() {
        const now = new Date();
        const dateStr = now.toISOString().slice(0, 19).replace('T', ' ');
        document.getElementById('liveClock').innerText = dateStr;
    }
    setInterval(updateClock, 1000);
    updateClock(); // Initial call
    </script>
    <div class="live-clock" id="liveClock">Loading...</div>
    """
    html(clock_js, height=50)

# Sidebar for file upload and filters
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # File uploader
    uploaded_file = st.file_uploader("📁 Upload CSV File", type=['csv'])
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    st.markdown("---")
    st.markdown("### 📅 Date Range Selection")
    
    # Date range selection
    date_option = st.selectbox(
        "Select Time Period",
        ["All Data", "Today", "Yesterday", "Last 7 Days", "Last 30 Days", 
         "This Month", "Last Month", "Custom Range"]
    )
    
    date_range_start = None
    date_range_end = None
    
    if date_option == "Custom Range":
        col1, col2 = st.columns(2)
        with col1:
            date_range_start = st.date_input("Start Date", datetime.now() - timedelta(days=7))
        with col2:
            date_range_end = st.date_input("End Date", datetime.now())
    elif date_option != "All Data":
        now = datetime.now()
        if date_option == "Today":
            date_range_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            date_range_end = now
        elif date_option == "Yesterday":
            date_range_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            date_range_end = (now - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
        elif date_option == "Last 7 Days":
            date_range_start = now - timedelta(days=7)
            date_range_end = now
        elif date_option == "Last 30 Days":
            date_range_start = now - timedelta(days=30)
            date_range_end = now
        elif date_option == "This Month" or date_option == "Last Month":
            start_date, end_date = get_month_range(date_option)
            date_range_start = start_date
            date_range_end = end_date
    
    st.markdown("---")
    st.markdown("### 🔍 Device Filters")
    
    # Initialize session state for filters
    if 'selected_devices' not in st.session_state:
        st.session_state.selected_devices = []
    if 'search_device' not in st.session_state:
        st.session_state.search_device = ""

# Main content area
if uploaded_file is None:
    st.info("👈 Please upload a CSV file using the sidebar to get started.")
    st.markdown("### 📋 Expected CSV Format")
    st.write("""
    Your CSV file should contain at least these columns:
    - **Device Name**: Name of the device
    - **Record Time**: Timestamp in DD-MM-YYYY format
    - **Type**: Contains 'encoding', 'online', or 'offline'
    
    Example format:
    ```
    Device Name,Record Time,Type
    Device_01,01-11-2024 10:00:00,encoding_online
    Device_01,01-11-2024 10:30:00,encoding_offline
    Device_02,01-11-2024 11:00:00,encoding_online
    ```
    """)
else:
    try:
        # Process the uploaded file
        with st.spinner("Processing CSV file..."):
            df, original_stats, processed_devices = process_csv(uploaded_file, date_range_start, date_range_end)
        
        if df is not None and not df.empty:
            # Calculate summary and downtime
            with st.spinner("Calculating downtime analysis..."):
                summary, df_downtime = calculate_downtime(df)
            
            # Get unique devices for filter
            all_devices = sorted(df['Device Name'].unique().tolist())
            
            # Display info cards
            st.markdown('<h2 class="sub-header">📈 File Statistics</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">📊</div>
                    <div class="metric-value">{original_stats['total_records']:,}</div>
                    <div class="metric-label">Total Records</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">📋</div>
                    <div class="metric-value">{original_stats['total_columns']}</div>
                    <div class="metric-label">Total Columns</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">🔧</div>
                    <div class="metric-value">{original_stats['total_devices']:,}</div>
                    <div class="metric-label">Total Devices</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">⚙️</div>
                    <div class="metric-value active-devices">{processed_devices:,}</div>
                    <div class="metric-label">Devices with Encoding Data</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Filters section
            st.markdown('<h2 class="sub-header">🔍 Device Filters</h2>', unsafe_allow_html=True)
            
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                
                with col1:
                    # Device search
                    search_device = st.text_input("🔍 Search Device", 
                                                  value=st.session_state.search_device,
                                                  placeholder="Type to search devices...")
                    st.session_state.search_device = search_device
                
                with col2:
                    # Select all checkbox
                    select_all = st.checkbox("✅ Select All Devices", value=True)
                
                with col3:
                    # Show count
                    st.metric("Devices", len(all_devices))
                
                with col4:
                    # Clear filter button
                    if st.button("🗑️ Clear Filter", use_container_width=True):
                        st.session_state.selected_devices = all_devices
                        st.session_state.search_device = ""
                        st.rerun()
                
                # Device multi-select with search
                if search_device:
                    filtered_devices = [d for d in all_devices if search_device.lower() in d.lower()]
                else:
                    filtered_devices = all_devices
                
                if select_all:
                    selected_devices = st.multiselect(
                        "Select Devices",
                        options=filtered_devices,
                        default=filtered_devices,
                        key="device_multiselect"
                    )
                else:
                    selected_devices = st.multiselect(
                        "Select Devices",
                        options=filtered_devices,
                        default=st.session_state.selected_devices if st.session_state.selected_devices else [],
                        key="device_multiselect"
                    )
                
                st.session_state.selected_devices = selected_devices
                
                # Show filter stats
                filtered_device_count = len(selected_devices) if selected_devices else len(all_devices)
                st.caption(f"📊 Showing **{filtered_device_count}** of **{len(all_devices)}** devices")
            
            # Apply filters to summary
            filtered_summary = summary.copy()
            filtered_downtime = df_downtime.copy()
            
            if selected_devices:
                filtered_summary = filtered_summary[filtered_summary['Device'].isin(selected_devices)]
                filtered_downtime = filtered_downtime[filtered_downtime['Device'].isin(selected_devices)]
            else:
                # If no selection, show all
                selected_devices = all_devices
            
            # Create tabs
            tab1, tab2 = st.tabs(["📊 Overview", "📋 Details"])
            
            with tab1:
                # Overview tab
                st.markdown('<h2 class="sub-header">📊 Device Overview Summary</h2>', unsafe_allow_html=True)
                
                # Export button for overview
                col1, col2 = st.columns([3, 1])
                with col1:
                    overview_count = len(filtered_summary)
                    st.markdown(f"**Device Overview** <span class='stats-badge'>{overview_count} devices</span>", unsafe_allow_html=True)
                with col2:
                    if not filtered_summary.empty:
                        # Prepare data for download
                        download_summary = filtered_summary.copy()
                        download_summary['Current_Status'] = download_summary['Current_Status'].str.replace('🔴 ', 'Offline - ').str.replace('✔️ ', 'Online - ')
                        
                        # Export to Excel
                        excel_data = to_excel({"Overview": download_summary})
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label="📥 Export Overview",
                            data=excel_data,
                            file_name=f"device_overview_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            type="primary"
                        )
                
                # Display overview table
                display_summary = filtered_summary.copy()
                display_summary['Last_Offline_Time'] = display_summary['Last_Offline_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(
                    display_summary,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Device": st.column_config.TextColumn("Device", width="medium"),
                        "Current_Status": st.column_config.TextColumn("Current Status", width="small"),
                        "Ongoing_Count": st.column_config.NumberColumn("Ongoing Events", width="small"),
                        "Last_Offline_Time": st.column_config.TextColumn("Last Offline Time", width="medium"),
                        "Total_DownTime_Events": st.column_config.NumberColumn("Total Events", width="small"),
                        "Current_Downtime_Duration": st.column_config.TextColumn("Current Downtime", width="medium"),
                        "Total_Downtime_Duration": st.column_config.TextColumn("Total Downtime", width="medium"),
                    }
                )
            
            with tab2:
                # Details tab
                st.markdown('<h2 class="sub-header">📋 Downtime Event Details</h2>', unsafe_allow_html=True)
                
                if not filtered_downtime.empty:
                    # Export button for details
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        details_count = len(filtered_downtime)
                        st.markdown(f"**Downtime Events** <span class='stats-badge'>{details_count} events</span>", unsafe_allow_html=True)
                    with col2:
                        # Prepare data for download
                        download_downtime = filtered_downtime.copy()
                        download_downtime['Downtime_Status'] = download_downtime['Downtime_Status'].str.replace('🔴 ', 'Ongoing - ').str.replace('✔️ ', 'Completed - ')
                        
                        # Export to Excel
                        excel_data = to_excel({"Details": download_downtime})
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label="📥 Export Details",
                            data=excel_data,
                            file_name=f"downtime_details_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            type="primary"
                        )
                    
                    # Display details table
                    display_downtime = filtered_downtime.copy()
                    display_downtime['Offline_Time'] = display_downtime['Offline_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    display_downtime['Online_Time'] = display_downtime['Online_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    st.dataframe(
                        display_downtime,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Device": st.column_config.TextColumn("Device", width="medium"),
                            "Offline_Time": st.column_config.TextColumn("Offline Time", width="medium"),
                            "Online_Time": st.column_config.TextColumn("Online Time", width="medium"),
                            "Downtime_Duration": st.column_config.TextColumn("Duration", width="medium"),
                            "Downtime_Status": st.column_config.TextColumn("Status", width="small"),
                        }
                    )
                else:
                    st.info("No downtime events found for the selected filters.")
            
            # Export All Data button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if not summary.empty:
                    # Prepare all data for export
                    all_data = {
                        "Overview": summary.copy(),
                        "Details": df_downtime.copy(),
                        "Original_Stats": pd.DataFrame([original_stats])
                    }
                    
                    # Clean data for export
                    all_data["Overview"]['Current_Status'] = all_data["Overview"]['Current_Status'].str.replace('🔴 ', 'Offline - ').str.replace('✔️ ', 'Online - ')
                    all_data["Details"]['Downtime_Status'] = all_data["Details"]['Downtime_Status'].str.replace('🔴 ', 'Ongoing - ').str.replace('✔️ ', 'Completed - ')
                    
                    # Export to Excel
                    excel_data = to_excel(all_data)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="📁 Export All Data to Excel",
                        data=excel_data,
                        file_name=f"complete_downtime_analysis_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        type="secondary"
                    )
        
        else:
            st.warning("⚠️ No data found in the uploaded file or after applying date filters.")
            st.info("Please check your CSV file format or try a different date range.")
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("""
        **Common Issues:**
        1. Make sure your CSV has the required columns: 'Device Name', 'Record Time', 'Type'
        2. Check that timestamps are in a valid format (DD-MM-YYYY HH:MM:SS)
        3. Ensure the file is not corrupted
        4. Verify that 'encoding' text appears in the Type column
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>📊 Device Downtime Analyzer | Complete Overview & Details Reporting</p>
        <p><small>Processes encoding device offline/online events and calculates downtime durations</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
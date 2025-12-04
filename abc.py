import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# Format duration function
def format_duration(seconds):
    if pd.isna(seconds):
        return ""
    seconds = int(seconds)
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Process data function
def process_data(df, start_date=None, end_date=None, selected_devices=None):
    # Filter by date range if provided
    if start_date and end_date:
        df = df[(df['Record Time'] >= start_date) & (df['Record Time'] <= end_date + timedelta(days=1))]
    
    # Filter by selected devices if provided
    if selected_devices and len(selected_devices) > 0:
        df = df[df['Device Name'].isin(selected_devices)]
    
    # Process downtime
    current_time = pd.Timestamp.now()
    
    df_downtime = (
        df
        .assign(next_status=df.groupby('Device Name')['status'].shift(-1),
                next_time=df.groupby('Device Name')['Record Time'].shift(-1),
                prev_status=df.groupby('Device Name')['status'].shift(1))
        
        # Find all offline periods
        .loc[lambda x: x['status'] == 'offline']
        
        # Calculate downtime
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
    
    # Recalculate downtime
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
    summary['Current_Status'] = np.where(summary['Ongoing_Count'] > 0, '🔴 Offline', '✔️ Online')
    
    # Format downtime status with emojis
    df_downtime['Downtime_Status'] = np.where(df_downtime['Downtime_Status'] == 'Ongoing', '🔴 Ongoing', '✔️ Completed')
    
    return summary, df_downtime

# Streamlit App
def main():
    st.set_page_config(page_title="Device Downtime Report", layout="wide")
    
    st.title("📊 Device Downtime Report")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Report Controls")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Data preprocessing
                df['Record Time Format'] = pd.to_datetime(
                    df['Record Time'], 
                    dayfirst=True,
                    errors='coerce'
                )
                df['Record Time'] = df['Record Time Format']
                df = df.drop(columns=['Record Time Format'], errors='ignore')
                
                # Filter encoding records
                df = df[df['Type'].str.contains('encoding', case=False, na=False)]
                
                # Create status column
                df['status'] = 'unknown'
                df.loc[df['Type'].str.contains('online', case=False, na=False), 'status'] = 'online'
                df.loc[df['Type'].str.contains('offline', case=False, na=False), 'status'] = 'offline'
                df = df.drop(columns=['Type'], errors='ignore')
                
                df = df.sort_values(by=['Device Name', 'Record Time'], ascending=[True, True])
                
                st.success(f"File loaded successfully!")
                st.info(f"Total records: {len(df)}")
                
                # Date range selector
                st.subheader("Date Range Filter")
                min_date = df['Record Time'].min().date()
                max_date = df['Record Time'].max().date()
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
                with col2:
                    end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
                
                if start_date > end_date:
                    st.error("Start date must be before end date!")
                    start_date, end_date = min_date, max_date
                
                # Device filter
                st.subheader("Device Filter")
                all_devices = sorted(df['Device Name'].unique())
                selected_devices = st.multiselect(
                    "Select devices (empty = all)",
                    all_devices,
                    default=all_devices[:min(5, len(all_devices))]
                )
                
                # Add some space before the generate button
                st.write("")  # Empty line for spacing
                
                # Rerun button - moved to be below device filter
                rerun_report = st.button("Generate Report", type="primary", use_container_width=True)
                
                if rerun_report:
                    with st.spinner("Processing data..."):
                        summary, downtime = process_data(
                            df.copy(),
                            pd.to_datetime(start_date),
                            pd.to_datetime(end_date),
                            selected_devices
                        )
                    
                    # Store in session state
                    st.session_state.summary = summary
                    st.session_state.downtime = downtime
                    st.session_state.processed = True
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.warning("Please upload a CSV file to begin")
            st.session_state.processed = False
    
    # Main content area
    if 'processed' in st.session_state and st.session_state.processed:
        summary = st.session_state.summary
        downtime = st.session_state.downtime
        
        # Display table shapes and total devices at the top
        st.subheader("Report Overview")
        
        # Calculate online and offline counts
        total_online = len(summary[summary['Current_Status'] == '✔️ Online'])
        total_offline = len(summary[summary['Current_Status'] == '🔴 Offline'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Devices",
                value=len(summary),
                delta=None
            )
        with col2:
            st.metric(
                label="Total ✔️ Online",
                value=total_online,
                delta=None
            )
        with col3:
            st.metric(
                label="Total 🔴 Offline",
                value=total_offline,
                delta=None
            )
        
        # Divider
        st.divider()
        
        # Create display summary without Ongoing_Count column
        display_summary = summary[['Device', 'Current_Status', 'Last_Offline_Time', 
                                   'Total_DownTime_Events', 'Current_Downtime_Duration', 
                                   'Total_Downtime_Duration']]
        
        # Display summary table (without Ongoing_Count column)
        st.subheader(f"Summary Table {display_summary.shape[0]} rows × {display_summary.shape[1]} cols")
        st.dataframe(display_summary, use_container_width=True)
        
        # Download buttons for summary
        col1, col2 = st.columns(2)
        with col1:
            csv_summary = summary.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Summary CSV",
                data=csv_summary,
                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
                use_container_width=True
            )
        
        # Display downtime table
        st.subheader(f"Downtime Events {downtime.shape[0]} rows × {downtime.shape[1]} cols")
        st.dataframe(downtime, use_container_width=True)
        
        # Download buttons for downtime
        with col2:
            csv_downtime = downtime.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Downtime CSV",
                data=csv_downtime,
                file_name=f"downtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
                use_container_width=True
            )
    
    else:
        # Initial state or no file uploaded
        st.info("Please upload a CSV file using the sidebar controls to generate reports.")
        
        # Display sample of expected format
        with st.expander("Expected CSV Format"):
            st.code("""
Required columns:
- Record Time: Timestamp (DD-MM-YYYY HH:MM:SS)
- Device Name: Device identifier
- Type: Should contain 'encoding' and either 'online' or 'offline'

Example:
Record Time,Device Name,Type
01-11-2023 10:00:00,Device1,encoding online
01-11-2023 10:05:00,Device1,encoding offline
01-11-2023 10:10:00,Device1,encoding online
            """)

if __name__ == "__main__":
    main()

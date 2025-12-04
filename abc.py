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

# Process data function with error handling
def process_data(df, start_date=None, end_date=None, selected_devices=None):
    try:
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Filter by date range if provided
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            df = df[(df['Record Time'] >= start_date) & (df['Record Time'] <= end_date + timedelta(days=1))]
        
        # Filter by selected devices if provided
        if selected_devices and len(selected_devices) > 0:
            df = df[df['Device Name'].isin(selected_devices)]
        
        # Check if we have data after filtering
        if df.empty:
            # Return empty dataframes with correct structure
            empty_summary = pd.DataFrame(columns=[
                'Device', 'Current_Status', 'Last_Offline_Time', 
                'Total_DownTime_Events', 'Current_Downtime_Duration', 
                'Total_Downtime_Duration'
            ])
            empty_downtime = pd.DataFrame(columns=[
                'Device', 'Offline_Time', 'Online_Time', 
                'Downtime_Duration', 'Downtime_Status'
            ])
            return empty_summary, empty_downtime
        
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
        
        # Check if we have downtime data
        if df_downtime.empty:
            # Create empty summary with correct structure
            empty_summary = pd.DataFrame(columns=[
                'Device', 'Current_Status', 'Last_Offline_Time', 
                'Total_DownTime_Events', 'Current_Downtime_Duration', 
                'Total_Downtime_Duration'
            ])
            # Return empty dataframes
            return empty_summary, df_downtime
        
        # Fix misclassified records
        mask = (df_downtime['Online_Time'].notna()) & (df_downtime['Downtime_Status'] == 'Ongoing')
        df_downtime.loc[mask, 'Downtime_Status'] = 'Completed'
        
        # Recalculate downtime with error handling
        def recalculate_downtime(row):
            try:
                if row['Downtime_Status'] == 'Completed':
                    return row['Downtime_Seconds'] if not pd.isna(row['Downtime_Seconds']) else 0
                elif pd.notna(row['Online_Time']):
                    return (row['Online_Time'] - row['Offline_Time']).total_seconds()
                else:
                    return (current_time - row['Offline_Time']).total_seconds()
            except:
                return 0
        
        # Safely assign Downtime_Seconds
        df_downtime = df_downtime.copy()
        df_downtime.loc[:, 'Downtime_Seconds'] = df_downtime.apply(recalculate_downtime, axis=1)
        df_downtime.loc[:, 'Downtime_Duration'] = df_downtime['Downtime_Seconds'].apply(format_duration)
        
        # Create summary
        analysis_time = pd.Timestamp.now()
        
        # Check if we have data to group
        if df_downtime['Device'].nunique() == 0:
            empty_summary = pd.DataFrame(columns=[
                'Device', 'Current_Status', 'Last_Offline_Time', 
                'Total_DownTime_Events', 'Current_Downtime_Duration', 
                'Total_Downtime_Duration'
            ])
            return empty_summary, df_downtime
        
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
        
        # Handle case where summary might be empty
        if summary.empty:
            empty_summary = pd.DataFrame(columns=[
                'Device', 'Current_Status', 'Last_Offline_Time', 
                'Total_DownTime_Events', 'Current_Downtime_Duration', 
                'Total_Downtime_Duration'
            ])
            return empty_summary, df_downtime
        
        summary.columns = [
            'Device', 'Last_Offline_Time', 'Last_Online_Time',
            'Total_DownTime_Events', 'Total_Downtime_Seconds', 'Ongoing_Count'
        ]
        
        # Convert to appropriate types with error handling
        summary['Total_Downtime_Seconds'] = pd.to_numeric(summary['Total_Downtime_Seconds'], errors='coerce').fillna(0).round(0)
        summary['Current_Downtime_Seconds'] = np.where(
            summary['Ongoing_Count'] > 0,
            (analysis_time - pd.to_datetime(summary['Last_Offline_Time'])).dt.total_seconds().round(0),
            np.nan
        )
        
        # Ensure Total >= Current for ongoing devices
        ongoing_mask = summary['Ongoing_Count'] > 0
        summary.loc[ongoing_mask, 'Total_Downtime_Seconds'] = np.maximum(
            summary.loc[ongoing_mask, 'Total_Downtime_Seconds'],
            summary.loc[ongoing_mask, 'Current_Downtime_Seconds'].fillna(0)
        )
        
        # Format durations with error handling
        summary['Current_Downtime_Duration'] = summary['Current_Downtime_Seconds'].apply(lambda x: format_duration(x) if not pd.isna(x) else "")
        summary['Total_Downtime_Duration'] = summary['Total_Downtime_Seconds'].apply(format_duration)
        summary['Current_Status'] = np.where(summary['Ongoing_Count'] > 0, '🔴 Offline', '✔️ Online')
        
        # Format downtime status with emojis and remove Downtime_Seconds column
        df_downtime['Downtime_Status'] = np.where(df_downtime['Downtime_Status'] == 'Ongoing', '🔴 Ongoing', '✔️ Completed')
        
        # Select only needed columns for downtime table (remove Downtime_Seconds)
        df_downtime_display = df_downtime[['Device', 'Offline_Time', 'Online_Time', 'Downtime_Duration', 'Downtime_Status']]
        
        return summary, df_downtime_display
    
    except Exception as e:
        st.error(f"Error in process_data: {str(e)}")
        # Return empty dataframes on error
        empty_summary = pd.DataFrame(columns=[
            'Device', 'Current_Status', 'Last_Offline_Time', 
            'Total_DownTime_Events', 'Current_Downtime_Duration', 
            'Total_Downtime_Duration'
        ])
        empty_downtime = pd.DataFrame(columns=[
            'Device', 'Offline_Time', 'Online_Time', 
            'Downtime_Duration', 'Downtime_Status'
        ])
        return empty_summary, empty_downtime

# Streamlit App
def main():
    st.set_page_config(page_title="Device Downtime Report", layout="wide")
    
    st.title("📊 Device Downtime Report")
    
    # Initialize session state for status filters
    if 'summary_status_filter' not in st.session_state:
        st.session_state.summary_status_filter = "All"
    if 'downtime_status_filter' not in st.session_state:
        st.session_state.downtime_status_filter = "All"
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Report Controls")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], 
                                         on_change=lambda: st.session_state.update({
                                             "data_loaded": False, 
                                             "processed": False,
                                             "last_error": None,
                                             "summary_status_filter": "All",  # Reset filter on new file
                                             "downtime_status_filter": "All"  # Reset filter on new file
                                         }))
        
        # Auto-process when file is uploaded (shows all devices by default)
        if uploaded_file is not None and not st.session_state.data_loaded:
            try:
                with st.spinner("Loading and processing data..."):
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
                    
                    # Store the processed dataframe in session state
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    
                    # Auto-process with default settings (ALL devices by default)
                    min_date = df['Record Time'].min().date()
                    max_date = df['Record Time'].max().date()
                    all_devices = sorted(df['Device Name'].unique())
                    
                    # Process with ALL devices selected (empty list means all devices)
                    summary, downtime = process_data(
                        df.copy(),
                        pd.to_datetime(min_date),
                        pd.to_datetime(max_date),
                        []  # Empty list means ALL devices
                    )
                    
                    # Store in session state
                    st.session_state.summary = summary
                    st.session_state.downtime = downtime
                    st.session_state.processed = True
                    
                    st.success(f"File loaded and processed successfully!")
                    st.info(f"Total records: {len(df)}")
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.session_state.last_error = str(e)
        
        # Show filters and refresh button if data is loaded
        if 'df' in st.session_state and st.session_state.data_loaded:
            df = st.session_state.df
            
            # Date range selector with validation
            st.subheader("📅 Date Range Filter")
            min_date = df['Record Time'].min().date()
            max_date = df['Record Time'].max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
            
            # Validate date range
            if start_date > end_date:
                st.error("❌ Start date must be before end date!")
                # Auto-correct by swapping dates
                start_date, end_date = end_date, start_date
            
            # Device filter - NO default selection (empty means all)
            st.subheader("📱 Device Filter")
            all_devices = sorted(df['Device Name'].unique())
            
            # Multiselect with NO default devices selected (empty list)
            selected_devices = st.multiselect(
                "Select devices (empty = all)",
                all_devices,
                default=[]  # Empty list = ALL devices by default
            )
            
            # Add a separator before the refresh button
            st.write("---")
            
            # Refresh Report button
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Reset", use_container_width=True):
                    try:
                        with st.spinner("Updating report with new filters..."):
                            # If no devices selected, process ALL devices (empty list)
                            summary, downtime = process_data(
                                df.copy(),
                                pd.to_datetime(start_date),
                                pd.to_datetime(end_date),
                                selected_devices  # Empty list = all devices
                            )
                            
                            # Check if we got any data
                            if summary.empty and downtime.empty:
                                st.warning("⚠️ No data found for the selected filters. Please adjust your criteria.")
                            else:
                                # Update session state
                                st.session_state.summary = summary
                                st.session_state.downtime = downtime
                                st.session_state.processed = True
                                st.success("✅ Refreshed!")
                            
                    except Exception as e:
                        st.error(f"❌ Error updating report: {str(e)}")
                        st.session_state.last_error = str(e)
            
            with col2:
                if st.button("🗑️ reset", type="secondary", use_container_width=True):
                    # Clear all session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            
            # Show current filter info
            st.caption(f"📅 Date range: {start_date} to {end_date}")
            if selected_devices:
                st.caption(f"📱 Showing {len(selected_devices)} of {len(all_devices)} devices")
            else:
                st.caption(f"📱 Showing all {len(all_devices)} devices")
        
        elif uploaded_file is None:
            st.warning("📁 Please upload a CSV file to begin")
    
    # Main content area
    if 'processed' in st.session_state and st.session_state.processed:
        summary = st.session_state.summary
        downtime = st.session_state.downtime
        
        # Check if we have data
        if summary.empty and downtime.empty:
            st.warning("⚠️ No data found for the selected filters. Please adjust your criteria in the sidebar.")
        else:
            
            # Calculate online and offline counts
            total_online = len(summary[summary['Current_Status'] == '✔️ Online']) if not summary.empty else 0
            total_offline = len(summary[summary['Current_Status'] == '🔴 Offline']) if not summary.empty else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Total Devices",
                    value=len(summary) if not summary.empty else 0,
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
            
            if not summary.empty:
                # Create display summary without Ongoing_Count column
                display_summary = summary[['Device', 'Current_Status', 'Last_Offline_Time', 
                                           'Total_DownTime_Events', 'Current_Downtime_Duration', 
                                           'Total_Downtime_Duration']].copy()
                
                # Add dropdown filter for status at top right of summary table
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"📋 Summary Table ({display_summary.shape[0]} rows × {display_summary.shape[1]} cols)")
                
                with col2:
                    # Status filter dropdown for summary table
                    summary_status_options = ["All", "✔️ Online", "🔴 Offline"]
                    selected_summary_status = st.selectbox(
                        "Filter by Status",
                        options=summary_status_options,
                        index=summary_status_options.index(st.session_state.summary_status_filter) if st.session_state.summary_status_filter in summary_status_options else 0,
                        key="summary_status_filter_select",
                        label_visibility="collapsed"
                    )
                    
                    # Update session state
                    st.session_state.summary_status_filter = selected_summary_status
                
                # Apply status filter if not "All"
                if st.session_state.summary_status_filter != "All":
                    filtered_summary = display_summary[display_summary['Current_Status'] == st.session_state.summary_status_filter].copy()
                    st.caption(f"📊 Showing {len(filtered_summary)} devices with status: {st.session_state.summary_status_filter}")
                    st.dataframe(filtered_summary, use_container_width=True)
                else:
                    st.dataframe(display_summary, use_container_width=True)
                
                # Download button for summary - BELOW THE TABLE
                # Prepare data for download (filtered or full)
                if st.session_state.summary_status_filter != "All":
                    download_summary = display_summary[display_summary['Current_Status'] == st.session_state.summary_status_filter]
                else:
                    download_summary = display_summary
                
                csv_summary = download_summary.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Summary CSV",
                    data=csv_summary,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )
            else:
                st.info("ℹ️ No summary data available for the selected filters.")
            
            if not downtime.empty:
                # Add some spacing between tables
                st.write("")
                
                # Add dropdown filter for status at top right of downtime table
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"🔍 Downtime Events ({downtime.shape[0]} rows × {downtime.shape[1]} cols)")
                
                with col2:
                    # Status filter dropdown for downtime table
                    downtime_status_options = ["All", "✔️ Completed", "🔴 Ongoing"]
                    selected_downtime_status = st.selectbox(
                        "Filter by Status",
                        options=downtime_status_options,
                        index=downtime_status_options.index(st.session_state.downtime_status_filter) if st.session_state.downtime_status_filter in downtime_status_options else 0,
                        key="downtime_status_filter_select",
                        label_visibility="collapsed"
                    )
                    
                    # Update session state
                    st.session_state.downtime_status_filter = selected_downtime_status
                
                # Apply status filter if not "All"
                if st.session_state.downtime_status_filter != "All":
                    if st.session_state.downtime_status_filter == "✔️ Completed":
                        filtered_downtime = downtime[downtime['Downtime_Status'] == '✔️ Completed'].copy()
                    else:  # "🔴 Ongoing"
                        filtered_downtime = downtime[downtime['Downtime_Status'] == '🔴 Ongoing'].copy()
                    
                    st.caption(f"📊 Showing {len(filtered_downtime)} events with status: {st.session_state.downtime_status_filter}")
                    st.dataframe(filtered_downtime, use_container_width=True)
                else:
                    st.dataframe(downtime, use_container_width=True)
                
                # Download button for downtime - BELOW THE TABLE
                # Prepare data for download (filtered or full)
                if st.session_state.downtime_status_filter != "All":
                    if st.session_state.downtime_status_filter == "✔️ Completed":
                        download_downtime = downtime[downtime['Downtime_Status'] == '✔️ Completed']
                    else:  # "🔴 Ongoing"
                        download_downtime = downtime[downtime['Downtime_Status'] == '🔴 Ongoing']
                else:
                    download_downtime = downtime
                
                csv_downtime = download_downtime.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Downtime CSV",
                    data=csv_downtime,
                    file_name=f"downtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )
            else:
                st.info("ℹ️ No downtime events found for the selected filters.")
    
    else:
        # Initial state or no file uploaded
        st.info("👈 Please upload a CSV file using the sidebar controls to generate reports.")
        
        # Display sample of expected format
        with st.expander("📋 Expected CSV Format"):
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

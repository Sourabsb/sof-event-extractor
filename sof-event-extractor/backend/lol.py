import os
import io
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import shutil
import hashlib
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px

# Local modules
from sof_pipeline import (
	IngestedDoc,
	process_uploaded_files,
	extract_events_and_summary,
    calculate_laytime,
    LaytimeResult,
)

# Load .env if present
try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	pass


st.set_page_config(page_title="Laytime Calculator – SoF Intelligence", layout="wide")


# --- Utility Functions ---
def safe_float(value, default=0.0):
    """Safely convert a value to a float."""
    if value is None:
        return default
    try:
        # Attempt to remove commas for robust conversion
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return default

# --- State Management ---
def initialize_state():
    """Initialize session state variables."""
    if "docs_key" not in st.session_state:
        st.session_state.docs_key = 0
    if "summary_data" not in st.session_state:
        st.session_state.summary_data = {}
    if "events_df" not in st.session_state:
        st.session_state.events_df = pd.DataFrame()
    if "results" not in st.session_state:
        st.session_state.results = None

initialize_state()


# --- UI Helper Functions ---
def render_voyage_summary(summary_data: Dict) -> Dict:
    """Renders the voyage summary inputs and returns the current values."""
    st.subheader("Voyage Summary")
    
    updated_summary = {}
    
    col1, col2 = st.columns(2)
    with col1:
        updated_summary['CREATED FOR'] = st.text_input("Created For*", summary_data.get("CREATED FOR", ""), key="vessel_name")
        updated_summary['VOYAGE TO'] = st.text_input("Voyage To*", summary_data.get("VOYAGE TO", ""), key="voyage_to")
        updated_summary['PORT'] = st.text_input("Port*", summary_data.get("PORT", ""), key="port")
        updated_summary['DEMURRAGE'] = st.number_input("Demurrage ($/Day)*", value=safe_float(summary_data.get("DEMURRAGE")), min_value=0.0, format="%.2f", key="demurrage")
        updated_summary['LOAD/DISCH'] = st.number_input("Load/Disch (MT/Day)*", value=safe_float(summary_data.get("LOAD/DISCH")), min_value=0.0, format="%.2f", key="load_disch_rate")

    with col2:
        updated_summary['VOYAGE FROM'] = st.text_input("Voyage From*", summary_data.get("VOYAGE FROM", ""), key="voyage_from")
        updated_summary['CARGO'] = st.text_input("Cargo*", summary_data.get("CARGO", ""), key="cargo")
        updated_summary['OPERATION'] = st.selectbox("Operation*", ["Discharge", "Loading"], index=0 if summary_data.get("OPERATION", "Discharge") == "Discharge" else 1, key="operation")
        updated_summary['DISPATCH'] = st.number_input("Dispatch*", value=safe_float(summary_data.get("DISPATCH")), min_value=0.0, format="%.2f", key="dispatch")
        updated_summary['CARGO QTY'] = st.number_input("Cargo Qty (MT)*", value=safe_float(summary_data.get("CARGO QTY")), min_value=0.0, format="%.2f", key="cargo_qty")
        
    return updated_summary

def render_add_event_form():
    """Renders the form to add a new event."""
    st.subheader("Add New Event")
    with st.form("new_event_form", clear_on_submit=True):
        event_description = st.text_input("Event*")
        
        c1, c2 = st.columns(2)
        start_date = c1.date_input("Start Date*", datetime.now().date())
        start_time = c2.time_input("Start Time*", datetime.now().time())
        
        c3, c4 = st.columns(2)
        end_date = c3.date_input("End Date")
        end_time = c4.time_input("End Time")

        submitted = st.form_submit_button("Add Event")
        if submitted and event_description:
            start_dt = datetime.combine(start_date, start_time)
            end_dt = datetime.combine(end_date, end_time) if end_date and end_time else None
            
            new_event = {
                "filename": "Manual Entry",
                "source_page": "Manual",
                "event": event_description,
                "start_time_iso": start_dt,
                "end_time_iso": end_dt,
                "raw_line": "Manually added event"
            }
            
            new_df = pd.DataFrame([new_event])
            st.session_state.events_df = pd.concat([st.session_state.events_df, new_df], ignore_index=True)

def render_results(results: LaytimeResult):
    """Renders the calculation results."""
    st.subheader("Laytime Calculation Results")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Laytime Allowed", f"{results.laytime_allowed_days:.4f} Days")
    c2.metric("Laytime Consumed", f"{results.laytime_consumed_days:.4f} Days")
    
    if results.demurrage_due > 0:
        c3.metric("Result: Demurrage Due", f"${results.demurrage_due:,.2f}", delta_color="inverse")
    elif results.dispatch_due > 0:
        c3.metric("Result: Dispatch Due", f"${results.dispatch_due:,.2f}", delta_color="normal")
    else:
        c3.metric("Result: Balanced", "On Time")

    st.subheader("Final Statement of Facts")
    st.dataframe(results.events_df, use_container_width=True)

    # Add download buttons
    col1, col2 = st.columns(2)
    with col1:
        # Convert dataframe to CSV, ensuring datetime format is preserved
        csv_data = results.events_df.to_csv(index=False, date_format='%Y-%m-%d %H:%M:%S')
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name="laytime_calculation.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        # Convert dataframe to JSON
        json_data = results.events_df.to_json(orient="records", date_format="iso")
        st.download_button(
            label="Download as JSON",
            data=json_data,
            file_name="laytime_calculation.json",
            mime="application/json",
            use_container_width=True
        )

    with st.expander("View Calculation Log"):
        for entry in results.calculation_log:
            st.text(entry)

# --- Main App Logic ---
st.title("Laytime Calculator – SoF Intelligence")

# Environment check
if shutil.which("tesseract") is None:
	st.warning("Tesseract OCR is not installed. Scanned PDFs or photos may not yield text.")

# --- File Uploader & Processing ---
with st.expander("Upload & Extract Data", expanded=not st.session_state.get('summary_data')):
    uploaded_files = st.file_uploader(
        "Upload Statement of Facts (PDF, DOCX, TXT, JPG, PNG)",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.get('docs_key', 0)}"
    )

    if uploaded_files:
        if st.button("Extract Information"):
            with st.spinner("Analyzing documents... This may take a moment."):
                try:
                    docs = process_uploaded_files(uploaded_files)
                    if docs:
                        events_df, summary = extract_events_and_summary(docs, os.getenv("GROQ_API_KEY", ""))
                        st.session_state.events_df = events_df
                        st.session_state.summary_data = summary
                        st.session_state.results = None # Reset results on new data
                        st.session_state.docs_key = st.session_state.get('docs_key', 0) + 1
                        st.success("Information extracted successfully!")
                        st.rerun()
                    else:
                        st.error("No text could be extracted from the uploaded document(s).")
                except Exception as e:
                    st.error(f"An error occurred during extraction: {e}")


# --- Main Interactive UI ---
if st.session_state.get('summary_data'):
    st.header("Voyage & Laytime Parameters")
    st.session_state.summary_data = render_voyage_summary(st.session_state.summary_data)
    
    st.divider()
    
    render_add_event_form()

    st.divider()

    st.header("Statement of Facts Timeline")
    
    # Ensure datetime columns are of the correct type before rendering the editor
    if not st.session_state.events_df.empty:
        for col in ['start_time_iso', 'end_time_iso']:
            if col in st.session_state.events_df.columns:
                st.session_state.events_df[col] = pd.to_datetime(st.session_state.events_df[col], errors='coerce')

    st.session_state.events_df = st.data_editor(
        st.session_state.events_df, 
        use_container_width=True,
        num_rows="dynamic",
        key="events_editor",
        column_config={
            "start_time_iso": st.column_config.DatetimeColumn("Start Time", format="YYYY-MM-DD HH:mm", required=True),
            "end_time_iso": st.column_config.DatetimeColumn("End Time", format="YYYY-MM-DD HH:mm"),
            "event": st.column_config.TextColumn("Event", width="large"),
        }
    )
    
    st.divider()

    # --- Calculation Trigger ---
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Calculate Laytime", type="primary", use_container_width=True):
            with st.spinner("Calculating laytime..."):
                try:
                    results = calculate_laytime(st.session_state.summary_data, st.session_state.events_df)
                    st.session_state.results = results
                except Exception as e:
                    st.error(f"An error occurred during calculation: {e}")

# --- Display Results ---
if st.session_state.get('results'):
    render_results(st.session_state.results)
elif not st.session_state.get('summary_data'):
    st.info("Upload a Statement of Facts document to begin.")



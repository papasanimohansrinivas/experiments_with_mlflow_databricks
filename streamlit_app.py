import os
import json
import time
import pandas as pd
import numpy as np
import requests
import streamlit as st

st.set_page_config(page_title="Databricks Model Serving Client", layout="wide")

st.title("🚀 Databricks Model Serving — Streamlit Client")

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown(
        """
        This app calls a **Databricks Model Serving Endpoint** using the REST API:
        `POST /serving-endpoints/<endpoint-name>/invocations`.

        - Default payload format here is **`dataframe_split`** (best for PyFunc models).
        - You can also send **raw JSON** if your endpoint expects a custom schema.
        - Your Personal Access Token (PAT) is used only for this session and not stored server-side.
        """
    )

# --- Helpers ------------------------------------------------------------------

def _strip_trailing_slash(url: str) -> str:
    if url.endswith("/"):
        return url[:-1]
    return url

def build_invocations_url(host: str, endpoint_name: str) -> str:
    host = _strip_trailing_slash(host)
    return f"{host}/serving-endpoints/{endpoint_name}/invocations"

def create_tf_serving_json(data):
    # Keep behavior similar to original helper for non-DataFrame inputs
    if isinstance(data, dict):
        return {"inputs": {name: data[name].tolist() for name in data.keys()}}
    return data.tolist()

def build_dataframe_split_payload(df: pd.DataFrame) -> dict:
    return {"dataframe_split": df.to_dict(orient="split")}

def call_endpoint(host: str, token: str, endpoint_name: str, payload: dict, timeout_s: int = 120):
    url = build_invocations_url(host, endpoint_name)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data_json = json.dumps(payload, allow_nan=True)
    resp = requests.post(url, headers=headers, data=data_json, timeout=timeout_s)
    return resp

# --- Sidebar: Config -----------------------------------------------------------

st.sidebar.header("🔐 Databricks Connection")

use_secrets = st.sidebar.checkbox("Use st.secrets if available", value=("databricks" in st.secrets))

default_host = ""
default_token = ""
default_endpoint = ""

if use_secrets and "databricks" in st.secrets:
    default_host = st.secrets["databricks"].get("host", "")
    default_token = st.secrets["databricks"].get("token", "")
    default_endpoint = st.secrets.get("serving", {}).get("endpoint", "")
else:
    # Fallback to env vars if present
    default_host = os.environ.get("DATABRICKS_HOST", "")
    default_token = os.environ.get("DATABRICKS_TOKEN", "")
    # Endpoint name isn't standardized as env; leave blank by default
    default_endpoint = ""

host = st.sidebar.text_input("Databricks Host (https://...)", value=default_host, placeholder="https://<your-instance>.cloud.databricks.com")
token = st.sidebar.text_input("Personal Access Token (PAT)", value=default_token, type="password", placeholder="dapi...")
endpoint_name = st.sidebar.text_input("Serving Endpoint Name", value=default_endpoint, placeholder="your-endpoint-name")

st.sidebar.caption("Tip: Save these in `.streamlit/secrets.toml` for convenience.")

# --- Tabs for input modes ------------------------------------------------------

tab_text, tab_csv, tab_json = st.tabs(["📝 Single Text", "🗂️ Batch CSV", "🧩 Raw JSON"])

# ---- Single Text --------------------------------------------------------------
with tab_text:
    st.subheader("Single Text Inference (DataFrame mode)")

    col1, col2 = st.columns([2, 1])
    with col1:
        text_input = st.text_area("Input text", height=140, placeholder="Type text to send to the model...")
    with col2:
        df_col_name = st.text_input("DataFrame column name", value="text", help="Column name expected by your model code.")
        run_btn = st.button("▶️ Invoke Endpoint", use_container_width=True)

    if run_btn:
        if not host or not token or not endpoint_name:
            st.error("Please fill **Host**, **PAT**, and **Endpoint Name** in the sidebar.")
        elif not text_input.strip():
            st.warning("Please enter some text.")
        else:
            try:
                df = pd.DataFrame([{df_col_name: text_input}])
                payload = build_dataframe_split_payload(df)
                with st.spinner("Calling endpoint..."):
                    resp = call_endpoint(host, token, endpoint_name, payload)
                if resp.status_code == 200:
                    st.success("Success")
                    try:
                        st.json(resp.json())
                    except Exception:
                        st.code(resp.text)
                else:
                    st.error(f"Request failed: HTTP {resp.status_code}")
                    st.code(resp.text)
            except Exception as e:
                st.exception(e)

# ---- Batch CSV ----------------------------------------------------------------
with tab_csv:
    st.subheader("Batch CSV Inference (DataFrame mode)")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    batch_col = st.text_input("Text column in CSV", value="text")
    limit_rows = st.number_input("Max rows to send", min_value=1, value=32, step=1)
    colA, colB = st.columns(2)
    with colA:
        show_preview = st.checkbox("Show CSV preview", value=True)
    with colB:
        run_batch = st.button("▶️ Invoke Batch", use_container_width=True)

    if uploaded is not None:
        try:
            df_csv = pd.read_csv(uploaded)
            if show_preview:
                st.dataframe(df_csv.head(10))
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df_csv = None
    else:
        df_csv = None

    if run_batch:
        if not host or not token or not endpoint_name:
            st.error("Please fill **Host**, **PAT**, and **Endpoint Name** in the sidebar.")
        elif df_csv is None:
            st.warning("Please upload a CSV first.")
        elif batch_col not in df_csv.columns:
            st.error(f"Column '{batch_col}' not found in CSV.")
        else:
            try:
                df_send = df_csv[[batch_col]].head(int(limit_rows)).copy()
                payload = build_dataframe_split_payload(df_send)
                with st.spinner(f"Calling endpoint for {len(df_send)} rows..."):
                    resp = call_endpoint(host, token, endpoint_name, payload)
                if resp.status_code == 200:
                    st.success("Success")
                    try:
                        st.json(resp.json())
                    except Exception:
                        st.code(resp.text)
                else:
                    st.error(f"Request failed: HTTP {resp.status_code}")
                    st.code(resp.text)
            except Exception as e:
                st.exception(e)

# ---- Raw JSON -----------------------------------------------------------------
with tab_json:
    st.subheader("Raw JSON Payload (advanced)")

    st.markdown(
        "Use this when your endpoint expects a custom request format (e.g., `{'inputs': '...'}')."
    )
    example = {"dataframe_split": pd.DataFrame([{"text": "hello"}]).to_dict(orient="split")}
    default_json = json.dumps(example, indent=2)

    payload_text = st.text_area("Request JSON", value=default_json, height=220)
    send_raw_btn = st.button("▶️ Send Raw JSON", use_container_width=True, key="send_raw")

    if send_raw_btn:
        if not host or not token or not endpoint_name:
            st.error("Please fill **Host**, **PAT**, and **Endpoint Name** in the sidebar.")
        else:
            try:
                payload = json.loads(payload_text)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                payload = None

            if payload is not None:
                try:
                    with st.spinner("Calling endpoint..."):
                        resp = call_endpoint(host, token, endpoint_name, payload)
                    if resp.status_code == 200:
                        st.success("Success")
                        try:
                            st.json(resp.json())
                        except Exception:
                            st.code(resp.text)
                    else:
                        st.error(f"Request failed: HTTP {resp.status_code}")
                        st.code(resp.text)
                except Exception as e:
                    st.exception(e)

# --- Footer --------------------------------------------------------------------
st.caption("Built with ❤️ for Databricks Model Serving")
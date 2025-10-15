import os
import json
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Databricks Model Serving Client", layout="wide")

st.title("🚀 Databricks Model Serving — Streamlit Client (Hardcoded)")

with st.expander("ℹ️ About", expanded=False):
    st.markdown(
        """
        This app calls a **Databricks Model Serving Endpoint** using a **hardcoded** host, token, and endpoint name.
        - Payload format defaults to **`dataframe_split`** (PyFunc-friendly).
        - Use the **Raw JSON** tab for custom payloads.
        """
    )

# ======= HARD-CODED CONFIG =======
DATABRICKS_HOST    = "https://dbc-6d3b551e-de77.cloud.databricks.com"   # <- change if needed
DATABRICKS_TOKEN   = "dapi24ee8bd1cb5f03c2bfe11ef801c3c00a"                        # <- paste your PAT or use st.secrets/env
SERVING_ENDPOINT   = "mosaicbert-mlm-endpoint-11"                        # <- change if needed
# =================================

# Optional overrides via secrets or env (no UI inputs involved)
DATABRICKS_HOST  = st.secrets.get("databricks", {}).get("host", os.getenv("DATABRICKS_HOST", DATABRICKS_HOST))
DATABRICKS_TOKEN = st.secrets.get("databricks", {}).get("token", os.getenv("DATABRICKS_TOKEN", DATABRICKS_TOKEN))
SERVING_ENDPOINT = st.secrets.get("serving", {}).get("endpoint", os.getenv("DATABRICKS_ENDPOINT", SERVING_ENDPOINT))

# --- Helpers ------------------------------------------------------------------
def _strip_trailing_slash(url: str) -> str:
    return url[:-1] if url.endswith("/") else url

def build_invocations_url(host: str, endpoint_name: str) -> str:
    host = _strip_trailing_slash(host)
    return f"{host}/serving-endpoints/{endpoint_name}/invocations"

def build_dataframe_split_payload(df: pd.DataFrame) -> dict:
    return {"dataframe_split": df.to_dict(orient="split")}

def call_endpoint(host: str, token: str, endpoint_name: str, payload: dict, timeout_s: int = 120):
    url = build_invocations_url(host, endpoint_name)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    data_json = json.dumps(payload, allow_nan=True)
    resp = requests.post(url, headers=headers, data=data_json, timeout=timeout_s)
    return resp

# -- Connection status (non-editable) ------------------------------------------
cc1, cc2, cc3 = st.columns(3)
with cc1:
    st.metric("Host", DATABRICKS_HOST)
with cc2:
    st.metric("Endpoint", SERVING_ENDPOINT)
with cc3:
    masked = (DATABRICKS_TOKEN[:6] + "…" + DATABRICKS_TOKEN[-4:]) if len(DATABRICKS_TOKEN) >= 12 else "masked"
    st.metric("Token", masked)

if not DATABRICKS_HOST or not DATABRICKS_TOKEN or not SERVING_ENDPOINT:
    st.error("Hardcoded configuration is incomplete. Please set host, token, and endpoint in the source code (or via st.secrets / env).")
    st.stop()

# --- Tabs for input modes ------------------------------------------------------
tab_text, tab_csv, tab_json = st.tabs(["📝 Single Text", "🗂️ Batch CSV", "🧩 Raw JSON"])

# ---- Single Text --------------------------------------------------------------
with tab_text:
    st.subheader("Single Text Inference (DataFrame mode)")
    text_input = st.text_area("Input text", height=140, placeholder="Type text to send to the model...")
    df_col_name = st.text_input("DataFrame column name", value="text", help="Column name expected by your model code.")
    run_btn = st.button("▶️ Invoke Endpoint", use_container_width=True)

    if run_btn:
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            try:
                df = pd.DataFrame([{df_col_name: text_input}])
                payload = build_dataframe_split_payload(df)
                with st.spinner("Calling endpoint..."):
                    resp = call_endpoint(DATABRICKS_HOST, DATABRICKS_TOKEN, SERVING_ENDPOINT, payload)
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
    run_batch = st.button("▶️ Invoke Batch", use_container_width=True)

    df_csv = None
    if uploaded is not None:
        try:
            df_csv = pd.read_csv(uploaded)
            st.dataframe(df_csv.head(10))
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    if run_batch:
        if df_csv is None:
            st.warning("Please upload a CSV first.")
        elif batch_col not in df_csv.columns:
            st.error(f"Column '{batch_col}' not found in CSV.")
        else:
            try:
                df_send = df_csv[[batch_col]].head(int(limit_rows)).copy()
                payload = build_dataframe_split_payload(df_send)
                with st.spinner(f"Calling endpoint for {len(df_send)} rows..."):
                    resp = call_endpoint(DATABRICKS_HOST, DATABRICKS_TOKEN, SERVING_ENDPOINT, payload)
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
    example = {"dataframe_split": pd.DataFrame([{"text": "hello"}]).to_dict(orient="split")}
    default_json = json.dumps(example, indent=2)

    payload_text = st.text_area("Request JSON", value=default_json, height=220)
    send_raw_btn = st.button("▶️ Send Raw JSON", use_container_width=True, key="send_raw")

    if send_raw_btn:
        try:
            payload = json.loads(payload_text)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            payload = None

        if payload is not None:
            try:
                with st.spinner("Calling endpoint..."):
                    resp = call_endpoint(DATABRICKS_HOST, DATABRICKS_TOKEN, SERVING_ENDPOINT, payload)
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
st.caption("Built with ❤️ for Databricks Model Serving — Hardcoded config")

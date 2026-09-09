"""Page 12: Interactive Experiment Suite Runner."""
import streamlit as st
from streamlit_utils.theme import apply_theme
from experiments.experiment_suite import ExperimentSuiteRunner

apply_theme()
st.header("🧪 Automated Research Experiment Suite (28 Scenarios)")

if st.button("Execute All 28 Scenarios Batch Sweep"):
    with st.spinner("Executing simulation sweeps..."):
        runner = ExperimentSuiteRunner(output_dir="results/experiments")
        df_res = runner.run_all_scenarios()
        st.success("Successfully completed all 28 experimental scenarios!")
        st.dataframe(df_res, use_container_width=True)
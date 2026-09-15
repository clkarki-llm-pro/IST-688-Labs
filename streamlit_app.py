import streamlit as st

# Define the pages for the multi-page "Lab" application.
Lab1 = st.Page("Lab1.py", title="Lab 1", icon="🟠")
Lab2 = st.Page("Lab2.py", title="Lab 2", icon="🟡")
Lab3 = st.Page("Lab3.py", title="Lab 3", icon="🟠")
Lab4 = st.Page("Lab4.py", title="Lab 4", icon="🟡", default=True)

# Lab3 is the default landing page.
pg = st.navigation([Lab1, Lab2, Lab3, Lab4])
pg.run()
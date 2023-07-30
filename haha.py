import streamlit as st

# Display the initial image
image1 = st.image("input2.jpg")

# Create a button
button = st.button("Show Another Image")

# Check if the button is clicked
if button:
    # Display the second image
    image2 = st.image("input.png")

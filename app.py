# Import required libraries
import PIL
import streamlit as st
from ultralytics import YOLO

# Replace the relative path to your weight file
model_path = 'weights/best.pt'

# Setting page layout
st.set_page_config(
    page_title="Object Detection",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
    
)

# Creating sidebar
with st.sidebar:
    st.header("Image Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("Plant deficiency Detection")
st.caption('Upload a photo')
st.caption('click the :blue[Detect Objects] button and check the result.')
# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('Detect Objects'):
    res = model.predict(uploaded_image, conf=confidence)
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]

    # Flag to check if a healthy plant is detected
    healthy_plant_detected = False

    with col2:
        st.image(res_plotted, caption='Detected Image', use_column_width=True)

        if boxes:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(f"Detected object: {box.class_name} with confidence: {box.confidence}")
                    if box.class_name.lower() == 'healthy':
                        healthy_plant_detected = True  # Set flag if a healthy plant is detected

    # Display a success message if a healthy plant is detected
    if healthy_plant_detected:
        st.success("The plant is healthy!")

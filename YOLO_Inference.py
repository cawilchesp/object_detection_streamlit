import streamlit as st


def main():
    # setting page layout
    st.set_page_config(
        page_title="Interactive Interface for YOLOv8",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # main page heading
    st.title("YOLOv8 Inference")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        **👈 Select a demo from the sidebar** to see some examples
        of what Streamlit can do!
        ### Want to learn more?
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
            forums](https://discuss.streamlit.io)
        ### See more complex demos
        - Use a neural net to [analyze the Udacity Self-driving Car Image
            Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )

    # source_img = None
    # if source_selectbox == config.SOURCES_LIST[0]: # Image
    #     infer_uploaded_image(confidence, model)
    # elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    #     infer_uploaded_webcam(confidence, model)


if __name__ == '__main__':
    main()
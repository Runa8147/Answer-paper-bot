import streamlit as st
from PIL import Image
import io
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import numpy as np
import json

genai.configure(api_key="AIzaSyDiWTGFlAIL-U9Nx-ksxak1Aj8_GsCh-ew")
# Create the model
generation_config = {
  "temperature": 0.3,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
)


def analyze_image(image):
    """Extract text from image using OCR"""
    prompt = "analyze the image and perform ocr in the file. Nothing else in the response. You are an ocr AI model to analyze and scan answer paper of students. Be truthful and careful."
    response = model.generate_content([image, prompt])
    return response.text


def get_answer_key_content(image_file=None, text_content=None):
    """Extract content from either image or text answer key"""
    if image_file:
        return analyze_image(image_file)
    return text_content


def compare_with_answer_key(student_response, answer_key):
    prompt = f"""
    Analyze the student's response compared to the answer key and provide the results in JSON format with the following structure:
    {{
        "question_scores": [
            {{"question": 1, "score": X, "max_points": Y}},
            // ... for each question
        ],
        "total_score": X,
        "max_total": Y,
        "feedback": {{
            "strengths": ["strength1", "strength2"],
            "improvements": ["improvement1", "improvement2"]
        }}
    }}
    
    Be lenient in grading - don't deduct points for minor spelling or syntax errors.
    Award partial credit for answers that show understanding but aren't exact matches.
    
    Student Response:
    {student_response}
    
    Answer Key:
    {answer_key}
    """
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except:
        return {
            "question_scores": [],
            "total_score": 0,
            "max_total": 100,
            "feedback": {
                "strengths": ["Unable to parse response"],
                "improvements": ["Unable to parse response"]
            }
        }


# Streamlit app configuration
st.title("Answer Paper Correction Bot")

# Create two columns for the main layout
col1, col2 = st.columns([3, 2])

with col1:
    # Student Answer Input
    st.subheader("Student Answer")
    input_method = st.radio("Select student answer input method", ("Upload from file", "Capture from camera"))

    student_image = None
    if input_method == "Upload from file":
        uploaded_file = st.file_uploader("Upload student answer image:", type=["jpg", "jpeg", "png", "svg"])
        if uploaded_file:
            student_image = Image.open(uploaded_file)
            st.image(student_image, caption="Student Answer", use_column_width=True)

    elif input_method == "Capture from camera":
        img_file_buffer = st.camera_input("Take a picture of student answer")
        if img_file_buffer:
            bytes_data = img_file_buffer.getvalue()
            student_image = Image.open(io.BytesIO(bytes_data))
            st.image(student_image, caption="Student Answer", use_column_width=True)

    # Answer Key Input
    st.subheader("Answer Key")
    answer_key_method = st.radio("Answer Key Input Method", ("Text Input", "Image Upload"))

    answer_key = None
    answer_key_image = None

    if answer_key_method == "Text Input":
        answer_key = st.text_area("Enter Answer Key", height=200)
    else:
        answer_key_file = st.file_uploader("Upload Answer Key Image", type=["jpg", "jpeg", "png", "svg"])
        if answer_key_file:
            answer_key_image = Image.open(answer_key_file)
            st.image(answer_key_image, caption="Answer Key", use_column_width=True)

with col2:
    st.subheader("Analysis Results")
    student_response = None
    answer_key_content = None
    results = None

    # Analyze Paper Button
    if st.button("Analyze Paper"):
        if student_image and (answer_key or answer_key_image):
            with st.spinner("Analyzing student answer..."):
                student_response = analyze_image(student_image)
                
                # Get answer key content
                if answer_key_image:
                    answer_key_content = analyze_image(answer_key_image)
                else:
                    answer_key_content = answer_key
                
                # Compare responses
                results = compare_with_answer_key(student_response, answer_key_content)
                st.success("Analysis Complete! You can now create a report.")
                st.session_state["results"] = results
                st.session_state["student_response"] = student_response
                st.session_state["answer_key_content"] = answer_key_content
        else:
            st.error("Please provide both student answer and answer key.")

    # Create Report Button
    if st.button("Create Report"):
        if "results" in st.session_state:
            results = st.session_state["results"]
            student_response = st.session_state["student_response"]
            answer_key_content = st.session_state["answer_key_content"]

            # Display total score and pass/fail status
            total_score = results["total_score"]
            max_total = results["max_total"]
            passing_score = max_total * 0.4  # 40% passing threshold

            st.metric(
                "Total Score",
                f"{total_score}/{max_total}",
                delta="PASS" if total_score >= passing_score else "FAIL"
            )

            # Simple progress bar for score visualization
            st.progress(total_score / max_total)

            # Display extracted text for verification
            with st.expander("View Extracted Text"):
                st.subheader("Student's Answer (Extracted)")
                st.text(student_response)
                st.subheader("Answer Key (Extracted)")
                st.text(answer_key_content)

            # Display feedback
            st.subheader("Feedback")
            st.write("Strengths:")
            for strength in results["feedback"]["strengths"]:
                st.write(f"• {strength}")

            st.write("Areas for Improvement:")
            for improvement in results["feedback"]["improvements"]:
                st.write(f"• {improvement}")

            # Display question-wise scores in a compact format
            st.subheader("Question Scores")
            scores_data = ""
            for qs in results["question_scores"]:
                scores_data += f"Q{qs['question']}: {qs['score']}/{qs['max_points']}\n"
            st.text(scores_data)
        else:
            st.error("Please analyze the paper first.")

from my_agents import Summarizer,QnA,CodeGenerator
import streamlit as st
from extractor import extract_text_from_file
from CodeGenerator import execute_generated_code
from data_analysis import DataInsights
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_together import ChatTogether

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    temperature=0.7,
    max_tokens=512,
    api_key="1882eed17b84bbee420d26f95a1342d5453e16fb1adfbe9caf161e1136143d7f"
)
summarize = Summarizer(llm=llm)
qna = QnA(llm=llm)
code = CodeGenerator(llm = llm)
datain = DataInsights(llm=llm)

# --- Streamlit UI ---
st.set_page_config(page_title="Document Q&A with Graphs")
st.title("📊 Document Q&A Agent with Code & Graph Generator")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "csv", "xlsx", "png", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner("Reading and summarizing document..."):
        doc_text = extract_text_from_file(uploaded_file)
        st.subheader("📄 Document Content Summary")
        summary_result = summarize.run(doc_text)
        st.write(summary_result)

    st.subheader("🧠 Ask Questions or Generate Graphs")
    st.markdown("""
    <style>
    div[data-baseweb="input"] > div {
        border: 2px solid #ddd !important;
        border-radius: 8px;
        padding: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

    user_question = st.text_input("📨 Ask a question (e.g., 'What is the summary?', 'Plot graph of sales')")

    if user_question:
        if any(word in user_question.lower() for word in ["plot", "graph", "visualize"]):
            with st.spinner("Generating and running code for graph..."):
                result = code.run(doc_text,user_question)
                st.markdown("### 📜 Answer")
                st.write(result.text)
                if result.code.strip():
                    st.markdown("### 🧾 Generated Code")
                    st.code(result.code, language="python")

                    output, fig, error = execute_generated_code(result.code)

                    if error:
                        st.error("⚠️ Error running code:")
                        st.text(error)
                    else:
                        if fig:
                            st.pyplot(fig)
                        if output.strip():
                            st.text(output.strip())

        elif any(word in user_question.lower() for word in ["code", "pragram", "programme"]):
            with st.spinner("Generating and running code for graph..."):
                result = code.run(doc_text,user_question)
                st.markdown("### 📜 Answer")
                st.write(result.text)
                if result.code.strip():
                    st.markdown("### 🧾 Generated Code")
                    st.code(result.code, language="python")
        



        else:
            with st.spinner("Answering question..."):
                result = qna.run(doc_text,user_question)
                st.markdown("### 📜 Answer")
                st.write(result)

    st.write("### Data Analysis")
    # Custom CSS for selectbox
    st.markdown("""
    <style>
    div[data-baseweb="select"] {
        border: 2px solid #4A90E2 !important;
        border-radius: 6px;
        padding: 4px;
    }
    </style>
""", unsafe_allow_html=True)



    options = st.selectbox("Select Analysis Feature", [
        "-- Select Feature --",
        "Descriptive Statistics",
        "Missing Value Analysis",
        "Correlation Matrix",
        "Outlier Detection",
        "Data Type Inference"
    ])

    if(options=="Descriptive Statistics") :

        result = datain.run_stats(doc_text)
        st.write(result)

    elif(options=="Missing Value Analysis") :

        
        result = datain.run_missing(doc_text)
        st.write(result)

    elif(options=="Correlation Matrix") :

        result = datain.run_correlation(doc_text)
        st.write(result.text)
        if result.code.strip():
            st.markdown("### 🧾 Generated Code")
            st.code(result.code, language="python")

            output, fig, error = execute_generated_code(result.code)

            if error:
                st.error("⚠️ Error running code:")
                st.text(error)
            else:
                if fig:
                    st.pyplot(fig)
                if output.strip():
                    st.text(output.strip())

    elif(options=="Outlier Detection") :
        result = datain.run_outlier(doc_text)
        st.write(result.text)
        if result.code.strip():
            st.markdown("### 🧾 Generated Code")
            st.code(result.code, language="python")

            output, fig, error = execute_generated_code(result.code)

            if error:
                st.error("⚠️ Error running code:")
                st.text(error)
            else:
                if fig:
                    st.pyplot(fig)
                if output.strip():
                    st.text(output.strip())

    elif(options=="Data Type Inference") :

        result = datain.run_outlier(doc_text)
        st.write(result)

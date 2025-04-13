import streamlit as st
import os
import traceback
from src.helper import (
    get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain,
    get_document_summary
)

# Set page configuration
st.set_page_config(
    page_title="AI-based Information Retrieval System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #4527A0;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #5E35B1;
    margin-bottom: 1rem;
}
.source-header {
    font-size: 1.2rem;
    color: #7E57C2;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.source-text {
    background-color: #F3E5F5;
    padding: 0.5rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
}
.summary-box {
    background-color: #E8EAF6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.chat-user {
    background-color: #E3F2FD;
    padding: 0.5rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
}
.chat-ai {
    background-color: #E0F7FA;
    padding: 0.5rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
}
.error-box {
    background-color: #FFEBEE;
    padding: 0.5rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
    color: #B71C1C;
}
.info-box {
    background-color: #E8F5E9;
    padding: 0.5rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
    color: #1B5E20;
}
</style>
""", unsafe_allow_html=True)


def display_sources(sources):
    """Display the source documents used for the answer"""
    if sources:
        st.markdown("<p class='source-header'>Sources:</p>", unsafe_allow_html=True)
        for i, source in enumerate(sources):
            st.markdown(f"<div class='source-text'><strong>Source {i+1}:</strong> {source.page_content[:300]}...</div>", unsafe_allow_html=True)


def user_input(user_question):
    """Process user questions and display responses with sources"""
    if st.session_state.conversation is None:
        st.error("Please process documents first.")
        return
    
    try:
        # Get response from conversation chain
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(f"<div class='chat-user'><strong>You:</strong> {message.content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-ai'><strong>AI:</strong> {message.content}</div>", unsafe_allow_html=True)
        
        # Display sources if available
        if 'source_documents' in response:
            display_sources(response['source_documents'])
    except Exception as e:
        st.markdown(f"<div class='error-box'><strong>Error:</strong> {str(e)}</div>", unsafe_allow_html=True)
        if st.checkbox("Show detailed error"):
            st.code(traceback.format_exc())


def main():
    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    
    # Main header
    st.markdown("<h1 class='main-header'>AI-based Information Retrieval System 🔍</h1>", unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    # Left column - Summary and Q&A
    with col1:
        # Summary section
        summary_expander = st.expander("Document Summary", expanded=True)
        with summary_expander:
            if st.session_state.summary:
                # Check if it's a fallback summary
                is_fallback = "[Note: This is an extractive summary created due to API limitations" in st.session_state.summary
                is_api_key_error = "API key is invalid or has expired" in st.session_state.summary
                
                if is_api_key_error:
                    st.warning("⚠️ API Key Issue Detected: Showing extractive summary instead of AI-generated summary")
                    st.info("This extractive summary contains key excerpts from your document. To get an AI-generated summary, please update your API key.")
                elif is_fallback:
                    st.warning("⚠️ API Quota Limit Reached: Showing extractive summary instead of AI-generated summary")
                    st.info("This extractive summary contains key excerpts from your document. To get an AI-generated summary, try again later when API quotas reset or try with a smaller document.")
                
                st.markdown(f"<div class='summary-box'>{st.session_state.summary}</div>", unsafe_allow_html=True)
            else:
                st.info("No summary available yet. Process documents and click 'Generate Summary' to create one.")
        
        # Question input
        st.markdown("<p class='sub-header'>Ask a question about your documents:</p>", unsafe_allow_html=True)
        user_question = st.text_input("Question", placeholder="Enter your question here...", label_visibility="collapsed")
        
        if user_question:
            user_input(user_question)
    
    # Right column - Document Management
    with col2:
        st.markdown("<p class='sub-header'>Document Management:</p>", unsafe_allow_html=True)
        
        # File uploader
        st.write("Upload your documents (PDF, DOCX, TXT):")
        uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True, type=["pdf", "docx", "txt"], label_visibility="collapsed")
        
        # Process and summarize buttons
        col_process, col_summarize = st.columns(2)
        process_btn = col_process.button("Process Documents")
        summarize_btn = col_summarize.button("Generate Summary")
        
        # Process documents when button is clicked
        if process_btn and uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    # Get text from documents
                    raw_text = get_pdf_text(uploaded_files)
                    st.session_state.raw_text = raw_text
                    
                    # Create text chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Create vector store
                    vector_store = get_vector_store(text_chunks)
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    
                    st.success("Documents processed successfully!")
                except Exception as e:
                    error_msg = str(e)
                    if "API key" in error_msg.lower() or "invalid" in error_msg.lower() or "expired" in error_msg.lower():
                        st.error("Google API Key Error")
                        st.markdown("""
                        ### 🔑 API Key Issue Detected
                        
                        Your Google API key appears to be invalid or has expired. Please follow these steps to fix it:
                        
                        1. **Get a new API key** from [Google AI Studio](https://ai.google.dev/)
                        2. **Update your `.env` file** with the new key:
                           ```
                           GOOGLE_API_KEY=your_new_api_key_here
                           ```
                        3. **Restart this application**
                        
                        For more information on Google API keys, see the [Google AI documentation](https://ai.google.dev/docs/api/get-api-key).
                        """)
                    else:
                        st.markdown(f"<div class='error-box'><strong>Error:</strong> {error_msg}</div>", unsafe_allow_html=True)
                    
                    if st.checkbox("Show detailed error for processing"):
                        st.code(traceback.format_exc())
        elif process_btn and not uploaded_files:
            st.warning("Please upload documents first.")
        
        # Generate summary when button is clicked
        if summarize_btn and st.session_state.raw_text:
            with st.spinner("Generating summary..."):
                try:
                    # Generate summary
                    summary = get_document_summary(st.session_state.raw_text)
                    
                    # Check if it's a fallback summary
                    is_fallback = "[Note: This is an extractive summary created due to API limitations" in summary
                    is_api_key_error = "API key is invalid or has expired" in summary
                    
                    if summary:
                        # Store the summary in session state
                        st.session_state.summary = summary
                        
                        if is_api_key_error:
                            st.error("Google API Key Error")
                            st.markdown("""
                            ### 🔑 API Key Issue Detected
                            
                            Your Google API key appears to be invalid or has expired. Please follow these steps to fix it:
                            
                            1. **Get a new API key** from [Google AI Studio](https://ai.google.dev/)
                            2. **Update your `.env` file** with the new key:
                               ```
                               GOOGLE_API_KEY=your_new_api_key_here
                               ```
                            3. **Restart this application**
                            
                            For more information on Google API keys, see the [Google AI documentation](https://ai.google.dev/docs/api/get-api-key).
                            """)
                            st.info("An extractive summary has been created as a fallback. Check the Document Summary section at the top left.")
                        elif is_fallback:
                            st.warning("API quota limit reached. An extractive summary has been generated instead.")
                            st.info("Check the Document Summary section at the top left to view the extractive summary.")
                        else:
                            st.success("Summary generated! Check the Document Summary section at the top left.")
                        
                        # Force a page refresh to show the summary
                        st.query_params.update(refresh=True)
                        st.rerun()
                    else:
                        st.error("Failed to generate summary. The summary returned was empty.")
                except Exception as e:
                    error_msg = str(e)
                    if "API key" in error_msg.lower() or "invalid" in error_msg.lower() or "expired" in error_msg.lower():
                        st.error("Google API Key Error")
                        st.markdown("""
                        ### 🔑 API Key Issue Detected
                        
                        Your Google API key appears to be invalid or has expired. Please follow these steps to fix it:
                        
                        1. **Get a new API key** from [Google AI Studio](https://ai.google.dev/)
                        2. **Update your `.env` file** with the new key:
                           ```
                           GOOGLE_API_KEY=your_new_api_key_here
                           ```
                        3. **Restart this application**
                        
                        For more information on Google API keys, see the [Google AI documentation](https://ai.google.dev/docs/api/get-api-key).
                        """)
                        
                        # Try to generate a fallback summary directly
                        try:
                            from src.helper import fallback_extractive_summary
                            fallback = fallback_extractive_summary(st.session_state.raw_text, error_message=error_msg)
                            if fallback:
                                st.session_state.summary = fallback
                                st.info("An extractive summary has been created. Check the Document Summary section.")
                                st.query_params.update(refresh=True)
                                st.rerun()
                        except Exception as fallback_error:
                            st.error(f"Could not create fallback summary: {str(fallback_error)}")
                    elif "429" in error_msg or "quota" in error_msg.lower():
                        st.error("Google API quota limit reached. Please try again later or reduce your document size.")
                        st.info("The system will attempt to create an extractive summary instead.")
                        
                        # Try to generate a fallback summary directly
                        try:
                            from src.helper import fallback_extractive_summary
                            fallback = fallback_extractive_summary(st.session_state.raw_text, error_message=error_msg)
                            if fallback:
                                st.session_state.summary = fallback
                                st.info("An extractive summary has been created. Check the Document Summary section.")
                                st.query_params.update(refresh=True)
                                st.rerun()
                        except Exception as fallback_error:
                            st.error(f"Could not create fallback summary: {str(fallback_error)}")
                    else:
                        st.markdown(f"<div class='error-box'><strong>Error:</strong> {error_msg}</div>", unsafe_allow_html=True)
                    
                    if st.checkbox("Show detailed error for summarization"):
                        st.code(traceback.format_exc())
        elif summarize_btn and not st.session_state.raw_text:
            st.warning("Please process documents first.")


if __name__ == "__main__":
    # Clear query params to avoid infinite refresh loops
    if "refresh" in st.query_params:
        st.query_params.clear()
    
    main()
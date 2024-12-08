import os
import re
import logging
import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Langchain imports
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit UI configuration (moved to the very top)
st.set_page_config(page_title="ChatGPT Blog Converter", page_icon="✍️")

def scrape_chatgpt_conversation(url):
    """
    Enhanced scraping function to extract ChatGPT conversation from various platforms.
    Uses multiple methods including requests and Selenium for dynamic content.
    """
    logging.info(f"Attempting to scrape URL: {url}")

    # Potential conversation selectors
    conversation_selectors = [
        
        # Specific platform selectors
        'div.prose',  # Some sharing platforms
        'div.whitespace-pre-wrap',  # Another potential selector
        'div[data-message-author]',  # Generic message container
        'div.markdown',  # New selector for markdown content
        'div.text-base'  # Another potential selector
    ]

    try:
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # Initialize WebDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Navigate to URL
        driver.get(url)
        
        # Wait for potential dynamic content loading
        time.sleep(5)  # Basic wait
        
        # Try different selectors with Selenium
        for selector in conversation_selectors:
            try:
                # Wait for elements to be present
                elements = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                )
                
                # Extract text from elements
                conversation_text = [
                    element.text.strip() 
                    for element in elements 
                    if element.text.strip()
                ]
                
                if conversation_text:
                    driver.quit()
                    logging.info("Successfully scraped content using Selenium")
                    return "\n\n".join(conversation_text)
            
            except Exception as inner_e:
                logging.warning(f"Selector {selector} failed: {inner_e}")
        
        driver.quit()
    
    except Exception as e:
        logging.error(f"Selenium scraping failed: {e}")

    raise ValueError("No conversation content found. Check the URL or sharing platform. Ensure the URL is publicly accessible and contains visible conversation text.")

def clean_conversation_text(text):
    """
    Clean and preprocess the scraped conversation text.
    """
    # Remove excessive whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove potential artifacts
    text = re.sub(r'(User|Assistant|ChatGPT):', '', text)
    
    # Additional cleaning
    text = text.strip()
    
    return text

def intelligent_chunking(text, max_chunks=20):
    """
    Intelligently chunk text based on its length.
    """
    total_length = len(text)
    
    # Adaptive chunking strategy
    if total_length < 5000:
        chunks = 5
    elif total_length < 20000:
        chunks = 10
    else:
        chunks = max_chunks

    # Calculate chunk size
    chunk_size = total_length // chunks
    
    # Split text into chunks
    words = text.split()
    chunked_text = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1

        if current_length >= chunk_size:
            chunked_text.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    # Add remaining chunk if not empty
    if current_chunk:
        chunked_text.append(" ".join(current_chunk))

    return chunked_text

def generate_blog_title(first_chunk):
    """
    Generate an engaging blog title based on the first chunk.
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        "Generate a catchy, SEO-friendly blog title that captures the essence of this conversation: {chunk}"
    )
    chain = prompt | model | StrOutputParser()
    
    try:
        title = chain.invoke({"chunk": first_chunk})
        return title
    except Exception as e:
        logging.error(f"Error generating title: {e}")
        return "Insights from a ChatGPT Conversation"

def generate_blog_sections(chunks):
    """
    Generate a single blog with context and continuity in a streaming manner.
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, streaming=True)
    output_parser = StrOutputParser()

    context = ""  # To store the entire blog content
    blog_content = ""  # The final content of the blog post

    # Create a single prompt that includes all chunks
    all_chunks = "\n\n".join(chunks)
    prompt = ChatPromptTemplate.from_template("""
        You are an expert content creator converting a ChatGPT conversation into a professional blog post.

        Conversation Content:
        {chunk}

        Guidelines:
        - Write a single, cohesive blog post
        - Maintain narrative flow and avoid repetition
        - Write in a professional yet conversational tone
        - Focus on key insights and actionable information
        - Use clear, engaging language
        - Format the output in markdown
    """)

    chain = prompt | model | output_parser

    # This will store the blog as it's being written
    for token in chain.stream({"chunk": all_chunks}):
        blog_content += token  # Append token to content
        yield blog_content  # This allows Streamlit to display it progressively

    # Return the full blog content
    return blog_content


# Streamlit UI and logic to display the content progressively
def main():
    # Attempt to load API key from environment variable
    default_api_key = os.getenv('OPENAI_API_KEY', '')

    # API Key input with environment variable support
    openai_api_key = st.text_input(
        "Enter OpenAI API Key", 
        value=default_api_key, 
        type="password"
    )

    # URL input
    url = st.text_input("Paste ChatGPT Conversation URL")

    # Generate button
    if st.button("Generate Blog"):
        if not openai_api_key or not url:
            st.warning("Please enter both API Key and URL")
            return

        try:
            # Set API Key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            openai.api_key = openai_api_key

            # Blog generation placeholders
            title_placeholder = st.empty()
            content_placeholder = st.empty()

            # Displaying progress and generating blog
            with title_placeholder.container():
                st.subheader("Blog Title")
                st.write("Generating...")

            full_blog_content = ""
            for chunk in generate_blog_sections(url):
                if not title_placeholder.empty():
                    title_placeholder.markdown(f"### {chunk}")
                    title_placeholder.empty()

                # Display the progressively generated content
                full_blog_content += chunk  # Accumulate content
                content_placeholder.markdown(full_blog_content, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating blog: {e}")


def generate_blog(url):
    """
    Main blog generation function.
    """
    try:
        # Scrape conversation
        conversation_text = scrape_chatgpt_conversation(url)
        
        # Clean conversation text
        cleaned_text = clean_conversation_text(conversation_text)
        
        # Chunk the text
        chunks = intelligent_chunking(cleaned_text)
        
        # Generate title from first chunk
        title = generate_blog_title(chunks[0])
        
        # Stream blog sections
        blog_content = ""
        for chunk in generate_blog_sections(chunks):
            blog_content += chunk
            yield chunk

        return title, blog_content

    except Exception as e:
        st.error(f"Blog generation error: {e}")
        return None, None

def main():
    # Attempt to load API key from environment variable
    default_api_key = os.getenv('OPENAI_API_KEY', '')
    
    # API Key input with environment variable support
    openai_api_key = st.text_input(
        "Enter OpenAI API Key", 
        value=default_api_key, 
        type="password"
    )
    
    # URL input
    url = st.text_input("Paste ChatGPT Conversation URL")
    
    # Generate button
    if st.button("Generate Blog"):
        if not openai_api_key or not url:
            st.warning("Please enter both API Key and URL")
            return

        try:
            # Set API Key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            openai.api_key = openai_api_key

            # Blog generation placeholders
            title_placeholder = st.empty()
            content_placeholder = st.empty()

            # Displaying progress and generating blog
            with title_placeholder.container():
                st.subheader("Blog Title")
                st.write("Generating...")
                
            for chunk in generate_blog(url):
                if not title_placeholder.empty():
                    title_placeholder.markdown(f"### {chunk}")
                    title_placeholder.empty()

                # Display the progressively generated content
                content_placeholder.markdown(chunk, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating blog: {e}")


if __name__ == "__main__":
    main()

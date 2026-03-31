import streamlit as st
from streamlit import session_state
import json
import os
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from collections import deque
import time
import pandas as pd
import matplotlib.pyplot as plt
import re
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def initialize_database(json_file_path="users.json"):
    if not os.path.exists(json_file_path):
        with open(json_file_path, "w") as f:
            json.dump({"users": []}, f)

def signup():
    st.title("Sign Up")
    with st.form("signup_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign Up")
        
        if submitted:
            if not email or not password:
                st.error("Please fill in all fields")
                return
                
            try:
                with open("users.json", "r") as f:
                    data = json.load(f)
                
                if any(user["email"] == email for user in data["users"]):
                    st.error("Email already exists")
                    return
                
                data["users"].append({
                    "email": email,
                    "password": password
                })
                
                with open("users.json", "w") as f:
                    json.dump(data, f)
                
                st.success("Account created successfully! Please login.")
                st.session_state.show_login = True
                
            except Exception as e:
                st.error(f"Error creating account: {str(e)}")

def login():
    st.title("Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            try:
                with open("users.json", "r") as f:
                    data = json.load(f)
                
                for user in data["users"]:
                    if user["email"] == email and user["password"] == password:
                        st.session_state.logged_in = True
                        st.session_state.user_info = user
                        st.success("Login successful!")
                        return
                
                st.error("Invalid email or password")
                
            except Exception as e:
                st.error(f"Error during login: {str(e)}")

class WebScraper:
    def __init__(self, max_depth: int = 2, max_pages: int = 50, max_workers: int = 5):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.max_workers = max_workers
        self.visited_urls = set()
        self.domain_blacklist = {
            'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
            'youtube.com', 'pinterest.com', 'tiktok.com', 'reddit.com'
        }
        self.content_types = {
            'text': ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'div', 'article', 'section'],
            'links': ['a'],
            'images': ['img'],
            'tables': ['table'],
            'lists': ['ul', 'ol', 'li']
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }

    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL is valid and belongs to the same domain."""
        try:
            parsed_url = urlparse(url)
            parsed_base = urlparse(base_domain)
            
            # Skip if URL is in blacklist
            if any(domain in parsed_url.netloc for domain in self.domain_blacklist):
                return False
                
            # Allow same domain and subdomains
            return (parsed_url.netloc == parsed_base.netloc or 
                   parsed_url.netloc.endswith('.' + parsed_base.netloc))
        except:
            return False

    def extract_content(self, url: str, max_retries: int = 3) -> Dict:
        """Extract structured content from a URL with retry logic."""
        for attempt in range(max_retries):
            try:
                with requests.Session() as session:
                    session.headers.update(self.headers)
                    response = session.get(url, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    content = {
                        'text': '',
                        'links': set(),
                        'images': set(),
                        'tables': [],
                        'lists': [],
                        'products': []
                    }
                    
                    # Extract text content
                    for tag in self.content_types['text']:
                        for element in soup.find_all(tag):
                            content['text'] += element.get_text(strip=True) + ' '
                    
                    # Enhanced product extraction
                    # Look for products in data attributes and specific product containers
                    product_elements = soup.find_all(['div', 'article', 'section'], 
                        class_=lambda x: x and any(cls in str(x).lower() for cls in ['product', 'item', 'card', 'grid', 'product-tile']))
                    
                    for product_elem in product_elements:
                        product_data = {}
                        
                        # Extract from data attributes first
                        if product_elem.has_attr('data-tileanalyticdata'):
                            try:
                                data = json.loads(product_elem['data-tileanalyticdata'])
                                product_data.update({
                                    'name': data.get('name', ''),
                                    'id': data.get('id', ''),
                                    'price': data.get('price', ''),
                                    'brand': data.get('brand', ''),
                                    'category': data.get('category', ''),
                                    'variant': data.get('variant', '')
                                })
                            except json.JSONDecodeError:
                                pass
                        
                        # Extract from data-tileanalyticdatag4 if available
                        if product_elem.has_attr('data-tileanalyticdatag4'):
                            try:
                                data = json.loads(product_elem['data-tileanalyticdatag4'])
                                product_data.update({
                                    'name': product_data.get('name', '') or data.get('item_name', ''),
                                    'id': product_data.get('id', '') or data.get('item_id', ''),
                                    'price': product_data.get('price', '') or data.get('price', ''),
                                    'brand': product_data.get('brand', '') or data.get('item_brand', ''),
                                    'category': product_data.get('category', '') or data.get('item_category', ''),
                                    'variant': product_data.get('variant', '') or data.get('item_variant', ''),
                                    'category2': data.get('item_category2', ''),
                                    'category3': data.get('item_category3', '')
                                })
                            except json.JSONDecodeError:
                                pass
                        
                        # Extract from HTML elements
                        # Product name and category from pdp-link
                        pdp_link = product_elem.find('div', class_='pdp-link')
                        if pdp_link:
                            category_elem = pdp_link.find('p', class_='mb-0')
                            if category_elem:
                                product_data['category'] = product_data.get('category', '') or category_elem.get_text(strip=True)
                            
                            name_elem = pdp_link.find('a', class_='link')
                            if name_elem:
                                product_data['name'] = product_data.get('name', '') or name_elem.get_text(strip=True)
                        
                        # Extract price from product-price-promotion
                        price_container = product_elem.find('div', class_='product-price-promotion')
                        if price_container:
                            # Look for price in sales span
                            price_elem = price_container.find('span', class_='sales')
                            if price_elem:
                                # Try to get price from content attribute
                                price = price_elem.find('span', attrs={'content': True})
                                if price:
                                    product_data['price'] = price['content']
                                else:
                                    # Try to get price from text
                                    price_text = price_elem.get_text(strip=True)
                                    price_match = re.search(r'₹?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', price_text)
                                    if price_match:
                                        product_data['price'] = price_match.group(1)
                        
                        # Extract images from main-image-slider
                        images = []
                        image_slider = product_elem.find('div', class_='main-image-slider')
                        if image_slider:
                            for img in image_slider.find_all('img'):
                                src = img.get('src') or img.get('data-src')
                                if src:
                                    full_url = urljoin(url, src)
                                    images.append(full_url)
                                    content['images'].add(full_url)
                        
                        # Extract color swatches
                        colors = []
                        swatches = product_elem.find('div', class_='color-swatches')
                        if swatches:
                            for swatch in swatches.find_all('img', class_='swatch'):
                                color_name = swatch.get('alt', '').split(',')[-1].strip()
                                if color_name:
                                    colors.append(color_name)
                        
                        # Extract product status (New, Sale, etc.)
                        status_tags = product_elem.find_all('span', class_=lambda x: x and any(cls in str(x).lower() for cls in ['new-tag', 'sale-tag']))
                        if status_tags:
                            product_data['status'] = [tag.get_text(strip=True) for tag in status_tags]
                        
                        # Extract quick view data
                        quickview = product_elem.find('a', class_='quickview')
                        if quickview and quickview.has_attr('href'):
                            product_data['quickview_url'] = urljoin(url, quickview['href'])
                        
                        # Extract wishlist data
                        wishlist = product_elem.find('a', class_='wishlistTile')
                        if wishlist and wishlist.has_attr('href'):
                            product_data['wishlist_url'] = urljoin(url, wishlist['href'])
                        
                        # Add extracted data to product
                        if images:
                            product_data['images'] = images
                        if colors:
                            product_data['colors'] = colors
                        
                        # Add product to list if we found any data
                        if product_data:
                            content['products'].append(product_data)
                    
                    # Extract links
                    for link in soup.find_all('a'):
                        href = link.get('href')
                        if href:
                            full_url = urljoin(url, href)
                            content['links'].add(full_url)
                    
                    # Extract images (non-product)
                    for img in soup.find_all('img'):
                        src = img.get('src') or img.get('data-src')
                        if src:
                            full_url = urljoin(url, src)
                            content['images'].add(full_url)
                    
                    # Extract tables
                    for table in soup.find_all('table'):
                        content['tables'].append(str(table))
                    
                    # Extract lists
                    for list_type in self.content_types['lists']:
                        for list_elem in soup.find_all(list_type):
                            content['lists'].append(str(list_elem))
                    
                    return content
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"All attempts failed for {url}: {e}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error extracting content from {url}: {e}")
                return None

    def scrape_website(self, start_url: str) -> Dict:
        """Scrape website content with improved logic."""
        base_domain = urlparse(start_url).netloc
        queue = deque([(start_url, 0)])
        all_content = {
            'text': '',
            'links': set(),
            'images': set(),
            'tables': [],
            'lists': []
        }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while queue and len(self.visited_urls) < self.max_pages:
                current_url, depth = queue.popleft()
                
                if current_url in self.visited_urls or depth > self.max_depth:
                    continue
                    
                self.visited_urls.add(current_url)
                
                # Submit URL for processing
                future = executor.submit(self.extract_content, current_url)
                content = future.result()
                
                if content:
                    # Merge content
                    all_content['text'] += content['text'] + ' '
                    all_content['links'].update(content['links'])
                    all_content['images'].update(content['images'])
                    all_content['tables'].extend(content['tables'])
                    all_content['lists'].extend(content['lists'])
                    
                    # Add new links to queue
                    for link in content['links']:
                        if (self.is_valid_url(link, base_domain) and 
                            link not in self.visited_urls and 
                            len(self.visited_urls) < self.max_pages):
                            queue.append((link, depth + 1))
        
        return all_content

class ContentAnalyzer:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
    def process_text(self, text: str) -> List[str]:
        """Process and chunk text for analysis."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=800
        )
        return text_splitter.split_text(text)
    
    def create_vector_store(self, chunks: List[str]):
        """Create vector store from text chunks."""
        vector_store = FAISS.from_texts(chunks, embedding=self.embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    
    def analyze_content(self, content: Dict, query: str = None) -> Dict:
        """Analyze scraped content using Gemini with enhanced detail extraction."""
        analysis = {
            'summary': '',
            'key_topics': [],
            'sentiment': '',
            'entities': [],
            'statistics': {},
            'products': content.get('products', []),  # Use directly scraped products
            'categories': [],
            'prices': [],
            'features': [],
            'specifications': [],
            'reviews': [],
            'ratings': [],
            'availability': [],
            'brands': [],
            'materials': [],
            'colors': [],
            'sizes': []
        }
        
        # Generate detailed analysis
        prompt = f"""
Analyze the following content{f" focusing on: {query}" if query else ""} and provide a comprehensive analysis.
Extract as much detail as possible about:
1. Products (name, description, price, features, specifications)
2. Categories and subcategories
3. Product features and specifications
4. Customer reviews and ratings
5. Product availability
6. Brands mentioned
7. Materials used
8. Available colors
9. Available sizes
10. Key topics and themes
11. Overall sentiment
12. Important entities

Respond with a valid JSON object that strictly follows this structure (ensure no comments or trailing commas):

{{
    "summary": "detailed summary here",
    "products": [
        {{
            "name": "product name",
            "description": "detailed description",
            "price": "price if available",
            "features": ["feature1", "feature2"],
            "specifications": ["spec1", "spec2"],
            "rating": "rating if available",
            "availability": "in stock/out of stock",
            "brand": "brand name",
            "material": "material type or null",
            "color": "color or null",
            "size": "size or null"
        }}
    ],
    "categories": ["main category", "subcategory1", "subcategory2"],
    "topics": ["topic1", "topic2"],
    "sentiment": "positive/negative/neutral",
    "entities": ["entity1", "entity2"],
    "brands": ["brand1", "brand2"],
    "materials": ["material1", "material2"],
    "colors": ["color1", "color2"],
    "sizes": ["size1", "size2"]
}}

Only return valid, parseable JSON.

Content:
\"\"\"
{content['text'][:10000].strip()}
\"\"\"
"""
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            response_text = response_text.replace("```json", "").replace("```","").strip("\n")
            # if response_text.endswith("```"):
            #     response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse the JSON response
            detailed_analysis = json.loads(response_text)
            
            # Update analysis with the parsed data
            analysis.update({
                'summary': detailed_analysis.get('summary', ''),
                'products': analysis['products'] + detailed_analysis.get('products', []),  # Combine scraped and analyzed products
                'categories': detailed_analysis.get('categories', []),
                'key_topics': detailed_analysis.get('topics', []),
                'sentiment': detailed_analysis.get('sentiment', ''),
                'entities': detailed_analysis.get('entities', []),
                'brands': detailed_analysis.get('brands', []),
                'materials': detailed_analysis.get('materials', []),
                'colors': detailed_analysis.get('colors', []),
                'sizes': detailed_analysis.get('sizes', [])
            })
            
            # Calculate basic statistics
            analysis['statistics'] = {
                'total_pages': len(content['links']),
                'total_images': len(content['images']),
                'total_tables': len(content['tables']),
                'total_lists': len(content['lists']),
                'text_length': len(content['text']),
                'total_products': len(analysis['products']),  # Use combined products count
                'total_categories': len(detailed_analysis.get('categories', [])),
                'total_brands': len(detailed_analysis.get('brands', [])),
                'total_materials': len(detailed_analysis.get('materials', [])),
                'total_colors': len(detailed_analysis.get('colors', [])),
                'total_sizes': len(detailed_analysis.get('sizes', []))
            }
            
            expected_stats = [
                'total_pages', 'total_images', 'total_tables', 'total_lists', 'text_length',
                'total_products', 'total_categories', 'total_brands', 'total_materials', 'total_colors', 'total_sizes'
            ]
            for stat in expected_stats:
                if stat not in analysis['statistics']:
                    analysis['statistics'][stat] = 0
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw response: {response.content if 'response' in locals() else 'No response'}")
            analysis['summary'] = "Error parsing analysis results. Please try again."
        except Exception as e:
            logger.error(f"Error in Gemini analysis: {e}")
            analysis['summary'] = "Error in content analysis. Please try again."
        
        return analysis

    def compare_websites(self, analysis1: Dict, analysis2: Dict, question: str = None):
        import io, base64
        def fig_to_base64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            return img_base64

        st.subheader("Website Comparison")
        comparison = {
            "summary": "",
            "statistics": {},
            "products": [],
            "categories": [],
            "brands": [],
            "materials": [],
            "colors": [],
            "sizes": [],
        }

        if question:
            prompt = f"""
            Compare the following two websites based on this question: {question}

            Website 1:
            {json.dumps(analysis1, indent=2)}

            Website 2:
            {json.dumps(analysis2, indent=2)}

            Provide a detailed comparison focusing on the question asked.
            """
            try:
                response = self.llm.invoke(prompt)
                summary = response.content
                comparison["summary"] = summary
                st.write("### Comparison Analysis")
                st.write(summary)
            except Exception as e:
                err_msg = f"Error generating comparison: {e}"
                comparison["summary"] = err_msg
                st.error(err_msg)

        st.write("### Statistics Comparison")
        metrics = ['Pages', 'Images', 'Tables', 'Lists', 'Products', 'Categories',
                'Brands', 'Materials', 'Colors', 'Sizes', 'Text Length']
        stat_keys = ['total_pages', 'total_images', 'total_tables', 'total_lists',
                    'total_products', 'total_categories', 'total_brands',
                    'total_materials', 'total_colors', 'total_sizes', 'text_length']
        data1 = [analysis1['statistics'].get(k, 0) for k in stat_keys]
        data2 = [analysis2['statistics'].get(k, 0) for k in stat_keys]
        comparison["statistics"] = {
            f"Website 1 - {m}": d1 for m, d1 in zip(metrics, data1)
        }
        comparison["statistics"].update({
            f"Website 2 - {m}": d2 for m, d2 in zip(metrics, data2)
        })

        comparison_df = pd.DataFrame({
            'Metric': metrics,
            'First Website': data1,
            'Second Website': data2
        })

        col1, col2 = st.columns(2)

        with col1:
            st.write("#### Content Metrics Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            comparison_df.plot(kind='bar', x='Metric', ax=ax)
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Count')
            ax.set_title('Content Statistics Comparison')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col2:
            st.write("#### Text Length Comparison")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            pd.DataFrame({
                'Website': ['First Website', 'Second Website'],
                'Text Length': [data1[-1], data2[-1]]
            }).plot(kind='bar', x='Website', y='Text Length', ax=ax2)
            ax2.set_ylabel('Characters')
            ax2.set_title('Text Length Comparison')
            st.pyplot(fig2)

        if analysis1['products'] or analysis2['products']:
            st.write("### Products Comparison")
            products1_df = pd.DataFrame(analysis1['products'])
            products2_df = pd.DataFrame(analysis2['products'])

            for prod in analysis1.get('products', []):
                prod["source"] = "Website 1"
                comparison["products"].append(prod)
            for prod in analysis2.get('products', []):
                prod["source"] = "Website 2"
                comparison["products"].append(prod)

            col1, col2 = st.columns(2)
            with col1:
                st.write("#### First Website Products")
                if not products1_df.empty:
                    st.dataframe(products1_df)
                else:
                    st.write("No products found")

            with col2:
                st.write("#### Second Website Products")
                if not products2_df.empty:
                    st.dataframe(products2_df)
                else:
                    st.write("No products found")

            if 'price' in products1_df.columns and 'price' in products2_df.columns:
                try:
                    products1_df['price'] = products1_df['price'].str.replace('₹', '').str.replace(',', '').str.strip()
                    products2_df['price'] = products2_df['price'].str.replace('₹', '').str.replace(',', '').str.strip()
                    products1_df['price'] = pd.to_numeric(products1_df['price'], errors='coerce')
                    products2_df['price'] = pd.to_numeric(products2_df['price'], errors='coerce')

                    if not products1_df['price'].isna().all() or not products2_df['price'].isna().all():
                        st.write("#### Price Distribution Comparison")
                        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                        products1_df['price'].dropna().hist(ax=ax1, bins=20)
                        ax1.set_xlabel('Price (₹)')
                        ax1.set_ylabel('Number of Products')
                        ax1.set_title('First Website Price Distribution')

                        products2_df['price'].dropna().hist(ax=ax2, bins=20)
                        ax2.set_xlabel('Price (₹)')
                        ax2.set_ylabel('Number of Products')
                        ax2.set_title('Second Website Price Distribution')

                        st.pyplot(fig3)

                        combined_prices = pd.concat([
                            products1_df[['price']],
                            products2_df[['price']]
                        ], ignore_index=True)
                        combined_prices = combined_prices['price'].dropna().tolist()

                        if combined_prices:
                            fig_combined, ax_comb = plt.subplots(figsize=(8, 4))
                            pd.Series(combined_prices).hist(ax=ax_comb, bins=20)
                            ax_comb.set_xlabel('Price')
                            ax_comb.set_ylabel('Count')
                            ax_comb.set_title('Product Price Distribution')
                            comparison["price_distribution_img"] = fig_to_base64(fig_combined)

                except Exception as e:
                    logger.error(f"Error creating price comparison: {e}")

        comparison["categories"] = list(set(analysis1.get("categories", []) + analysis2.get("categories", [])))
        comparison["brands"] = list(set(analysis1.get("brands", []) + analysis2.get("brands", [])))
        comparison["materials"] = list(set(analysis1.get("materials", []) + analysis2.get("materials", [])))
        comparison["colors"] = list(set(analysis1.get("colors", []) + analysis2.get("colors", [])))
        comparison["sizes"] = list(set(analysis1.get("sizes", []) + analysis2.get("sizes", [])))

        return comparison


def visualize_analysis(analysis: Dict, query: str = None):
    """Create detailed visualizations from analysis results."""
    st.subheader("Content Analysis")
    
    # Summary
    st.write("### Summary")
    st.write(analysis['summary'])
    
    # Products
    if analysis['products']:
        st.write("### Products")
        # Convert lists to strings in product data
        processed_products = []
        for product in analysis['products']:
            processed_product = {}
            for key, value in product.items():
                if isinstance(value, list):
                    processed_product[key] = ', '.join(str(v) for v in value)
                else:
                    processed_product[key] = str(value) if value is not None else ''
            processed_products.append(processed_product)
        
        products_df = pd.DataFrame(processed_products)
        if not products_df.empty:
            st.dataframe(products_df)
            
            # Product price distribution if prices are available
            if 'price' in products_df.columns:
                try:
                    # Clean price column
                    products_df['price'] = products_df['price'].str.replace('₹', '').str.replace(',', '').str.strip()
                    products_df['price'] = pd.to_numeric(products_df['price'], errors='coerce')
                    
                    if not products_df['price'].isna().all():
                        st.write("### Price Distribution")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        products_df['price'].hist(ax=ax, bins=20)
                        ax.set_xlabel('Price (₹)')
                        ax.set_ylabel('Number of Products')
                        ax.set_title('Product Price Distribution')
                        st.pyplot(fig)
                except Exception as e:
                    logger.error(f"Error creating price distribution: {e}")
    
    # Categories
    if analysis['categories']:
        st.write("### Categories")
        categories_df = pd.DataFrame({'Category': analysis['categories']})
        st.dataframe(categories_df)
        
        # Category distribution
        st.write("### Category Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        category_counts = categories_df['Category'].value_counts()
        ax.pie([1] * len(category_counts), labels=category_counts.index, autopct='')
        ax.set_title('Categories')
        st.pyplot(fig)
    
    # Brands
    if analysis['brands']:
        st.write("### Brands")
        brands_df = pd.DataFrame({'Brand': analysis['brands']})
        st.dataframe(brands_df)
        
        # Brand distribution
        st.write("### Brand Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        brand_counts = brands_df['Brand'].value_counts()
        ax.pie([1] * len(brand_counts), labels=brand_counts.index, autopct='')
        ax.set_title('Brands')
        st.pyplot(fig)
    
    # Materials
    if analysis['materials']:
        st.write("### Materials")
        # Convert list of materials to string if it's a list
        materials = analysis['materials']
        if isinstance(materials, list):
            materials = [', '.join(m) if isinstance(m, list) else str(m) for m in materials]
        materials_df = pd.DataFrame({'Material': materials})
        st.dataframe(materials_df)
        
        # Material distribution
        st.write("### Material Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        material_counts = materials_df['Material'].value_counts()
        ax.pie([1] * len(material_counts), labels=material_counts.index, autopct='')
        ax.set_title('Materials')
        st.pyplot(fig)
    
    # Colors
    if analysis['colors']:
        st.write("### Colors")
        # Convert list of colors to string if it's a list
        colors = analysis['colors']
        if isinstance(colors, list):
            colors = [', '.join(c) if isinstance(c, list) else str(c) for c in colors]
        colors_df = pd.DataFrame({'Color': colors})
        st.dataframe(colors_df)
        
        # Color distribution
        st.write("### Color Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        color_counts = colors_df['Color'].value_counts()
        ax.pie([1] * len(color_counts), labels=color_counts.index, autopct='')
        ax.set_title('Colors')
        st.pyplot(fig)
    
    # Sizes
    if analysis['sizes']:
        st.write("### Sizes")
        # Convert list of sizes to string if it's a list
        sizes = analysis['sizes']
        if isinstance(sizes, list):
            sizes = [', '.join(s) if isinstance(s, list) else str(s) for s in sizes]
        sizes_df = pd.DataFrame({'Size': sizes})
        st.dataframe(sizes_df)
        
        # Size distribution
        st.write("### Size Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        size_counts = sizes_df['Size'].value_counts()
        ax.pie([1] * len(size_counts), labels=size_counts.index, autopct='')
        ax.set_title('Sizes')
        st.pyplot(fig)
    
    # Statistics with improved visualization
    st.write("### Statistics")
    stats_data = {
        'Metric': ['Pages', 'Images', 'Tables', 'Lists', 'Products', 'Categories', 
                  'Brands', 'Materials', 'Colors', 'Sizes'],
        'Count': [
            analysis['statistics'].get('total_pages', 0),
            analysis['statistics'].get('total_images', 0),
            analysis['statistics'].get('total_tables', 0),
            analysis['statistics'].get('total_lists', 0),
            analysis['statistics'].get('total_products', 0),
            analysis['statistics'].get('total_categories', 0),
            analysis['statistics'].get('total_brands', 0),
            analysis['statistics'].get('total_materials', 0),
            analysis['statistics'].get('total_colors', 0),
            analysis['statistics'].get('total_sizes', 0)
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create a grid of charts
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Content Metrics")
        fig, ax = plt.subplots(figsize=(10, 6))
        stats_df.plot(kind='bar', x='Metric', y='Count', ax=ax)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Count')
        ax.set_title('Content Statistics')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.write("#### Text Length")
        st.metric("Total Characters", f"{analysis['statistics'].get('text_length', 0):,}")

def download_as_html(analysis, filename="analysis_results.html"):
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_base64

    html = [
        "<html><head><title>Web Analysis Results</title><style>body{font-family:Arial,sans-serif;margin:20px;}h1,h2{color:#333;}table{border-collapse:collapse;width:100%;margin-bottom:20px;}th,td{border:1px solid #ddd;padding:8px;}th{background:#f2f2f2;}section{margin-bottom:30px;}img{max-width:100%;height:auto;}</style></head><body>"
    ]
    html.append("<h1>Web Analysis Results</h1>")
    # Summary
    html.append("<section><h2>Summary</h2><p>{}</p></section>".format(analysis.get('summary', '')))
    print("Summary Appended\n**************************************************************************************")
    # Statistics
    stats = analysis.get('statistics', {})
    if stats:
        html.append("<section><h2>Statistics</h2><table><tr>{}</tr><tr>{}</tr></table></section>".format(
            ''.join(f"<th>{k.replace('_',' ').title()}</th>" for k in stats.keys()),
            ''.join(f"<td>{v}</td>" for v in stats.values())
        ))
    # Products
    products = analysis.get('products', [])
    if products:
        html.append("<section><h2>Products</h2><table><tr>{}</tr>".format(
            ''.join(f"<th>{k.title()}</th>" for k in products[0].keys())
        ))
        for prod in products:
            html.append("<tr>{}</tr>".format(''.join(f"<td>{str(v)}</td>" for v in prod.values())))
        html.append("</table></section>")
    # Categories
    categories = analysis.get('categories', [])
    if categories:
        html.append("<section><h2>Categories</h2><ul>{}</ul></section>".format(
            ''.join(f"<li>{c}</li>" for c in categories)
        ))
    # Brands
    brands = analysis.get('brands', [])
    if brands:
        html.append("<section><h2>Brands</h2><ul>{}</ul></section>".format(
            ''.join(f"<li>{b}</li>" for b in brands)
        ))
    # Materials
    materials = analysis.get('materials', [])
    if materials:
        html.append("<section><h2>Materials</h2><ul>{}</ul></section>".format(
            ''.join(f"<li>{m}</li>" for m in materials)
        ))
    # Colors
    colors = analysis.get('colors', [])
    if colors:
        html.append("<section><h2>Colors</h2><ul>{}</ul></section>".format(
            ''.join(f"<li>{c}</li>" for c in colors)
        ))
    # Sizes
    sizes = analysis.get('sizes', [])
    if sizes:
        html.append("<section><h2>Sizes</h2><ul>{}</ul></section>".format(
            ''.join(f"<li>{s}</li>" for s in sizes)
        ))
    # Price Distribution Graph (if available)
    if products and any('price' in p for p in products):
        import pandas as pd
        prices = [float(p['price']) for p in products if p.get('price') and str(p['price']).replace('.','',1).isdigit()]
        if prices:
            fig, ax = plt.subplots(figsize=(8,4))
            pd.Series(prices).hist(ax=ax, bins=20)
            ax.set_xlabel('Price')
            ax.set_ylabel('Count')
            ax.set_title('Product Price Distribution')
            img_base64 = fig_to_base64(fig)
            html.append(f'<section><h2>Price Distribution</h2><img src="data:image/png;base64,{img_base64}"/></section>')
    # Add more graphs as needed (e.g., category/brand pie charts)
    # End
    html.append("</body></html>")
    return ''.join(html)

def main():
    initialize_database()
    
    if not st.session_state.logged_in:
        st.sidebar.title("Authentication")
        if st.sidebar.button("Login"):
            st.session_state.show_login = True
        if st.sidebar.button("Sign Up"):
            st.session_state.show_signup = True
            
        if st.session_state.get("show_login"):
            login()
        elif st.session_state.get("show_signup"):
            signup()
        return
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Analysis", "Comparison", "Dashboard"])
    
    if page == "Home":
        st.title("Web Analysis Tool")
        st.write("Welcome to the Web Analysis Tool. Use the sidebar to navigate.")
        
    elif page == "Analysis":
        st.title("Web Analysis")
        url = st.text_input("Enter URL to analyze:")
        question = st.text_input("Enter comparison question (optional):", 
                               placeholder="e.g., 'Compare product offerings' or 'List all products'")
        
        if url:
            if st.button("Start Scraping"):
                with st.spinner("Scraping and analyzing content..."):
                    try:
                        scraper = WebScraper()
                        analyzer = ContentAnalyzer()
                        
                        content = scraper.scrape_website(url)
                        if not content or not content['text']:
                            st.error("Failed to scrape content from the provided URL. Please try again.")
                            return
                            
                        chunks = analyzer.process_text(content['text'])
                        if not chunks:
                            st.error("No content to analyze. Please try a different URL.")
                            return
                            
                        analysis = analyzer.analyze_content(content, question)
                        
                        # Display results
                        visualize_analysis(analysis)
                        
                        # Download options
                        st.subheader("Download Results")
                        
                        # Store analysis results in session state
                        st.session_state.analysis_results = analysis
                        
                        # Create download buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            html_content = download_as_html(analysis)
                            st.download_button(
                                label="Download as HTML",
                                data=html_content,
                                file_name="analysis_results.html",
                                mime="text/html"
                            )
                                
                    except Exception as e:
                        logger.error(f"Error during analysis: {e}")
                        st.error("An error occurred during analysis. Please try again.")
        else:
            st.info("Please enter a URL to analyze.")
            
    elif page == "Comparison":
        st.title("Website Comparison")
        url1 = st.text_input("Enter first URL:")
        url2 = st.text_input("Enter second URL:")
        question = st.text_input("Enter comparison question (optional):", 
                               placeholder="e.g., 'Compare product offerings' or 'List all products'")
        
        if url1 and url2:
            if st.button("Start Comparison"):
                with st.spinner("Scraping and comparing websites..."):
                    try:
                        scraper = WebScraper()
                        analyzer = ContentAnalyzer()
                        
                        # Scrape first website
                        content1 = scraper.scrape_website(url1)
                        if not content1 or not content1['text']:
                            st.error(f"Failed to scrape content from {url1}. Please try again.")
                            return
                            
                        # Scrape second website
                        content2 = scraper.scrape_website(url2)
                        if not content2 or not content2['text']:
                            st.error(f"Failed to scrape content from {url2}. Please try again.")
                            return
                        
                        # Process content
                        chunks1 = analyzer.process_text(content1['text'])
                        chunks2 = analyzer.process_text(content2['text'])
                        
                        if not chunks1 or not chunks2:
                            st.error("No content to analyze. Please try different URLs.")
                            return
                        
                        # Analyze content
                        analysis1 = analyzer.analyze_content(content1)
                        analysis2 = analyzer.analyze_content(content2)
                        
                        # Compare websites
                        comparison_results = analyzer.compare_websites(analysis1, analysis2, question)
                        print(comparison_results,"************************************************************************")
                        
                        # Store comparison results in session state
                        st.session_state.comparison_results = comparison_results
                        
                        # Download options
                        st.subheader("Download Comparison Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            html_content = download_as_html(comparison_results)
                            st.download_button(
                                label="Download as HTML",
                                data=html_content,
                                file_name="comparison_results.html",
                                mime="text/html"
                            )
                                
                    except Exception as e:
                        logger.error(f"Error during comparison: {e}")
                        st.error("An error occurred during comparison. Please try again.")
        else:
            st.info("Please enter both URLs to compare.")
                
    elif page == "Dashboard":
        st.title("User Dashboard")
        st.write(f"Welcome, {st.session_state.user_info['email']}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_info = None
            st.rerun()

if __name__ == "__main__":
    main()

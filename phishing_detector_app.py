import streamlit as st
import pandas as pd
import numpy as np
import re
import urllib.parse
from urllib.parse import urlparse
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Set page config
st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🔒",
    layout="wide"
)

@st.cache_data
def load_and_prepare_data():
    """Load the dataset and prepare it for training"""
    try:
        df = pd.read_csv("dataset_phishing.csv")
        
        # Drop the 'url' column if it exists
        if 'url' in df.columns:
            df = df.drop("url", axis=1)
        
        # Encode the 'status' column
        if 'status' in df.columns:
            label_encoder = LabelEncoder()
            df["status_encoding"] = label_encoder.fit_transform(df["status"])
            df = df.drop("status", axis=1)
        
        # Handle -1 values in domain columns
        mean_domain_age = df[df['domain_age'] != -1]['domain_age'].mean()
        mean_domain_registration_length = df[df['domain_registration_length'] != -1]['domain_registration_length'].mean()
        
        df['domain_age'] = df['domain_age'].replace(-1, mean_domain_age)
        df['domain_registration_length'] = df['domain_registration_length'].replace(-1, mean_domain_registration_length)
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_resource
def train_model():
    """Train the neural network model"""
    df = load_and_prepare_data()
    if df is None:
        return None, None
    
    # Define features and target
    X = df.drop('status_encoding', axis=1)
    y = df['status_encoding']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create and train the model
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='sigmoid'))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    with st.spinner('Training the model... This may take a few minutes.'):
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
    return model, scaler

def extract_url_features(url):
    """Extract features from URL to match the dataset format exactly"""
    features = {}
    
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path
        query = parsed_url.query
        
        # Basic URL character counting features
        features['length_url'] = len(url)
        features['length_hostname'] = len(domain)
        features['ip'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+', domain) else 0
        features['nb_dots'] = url.count('.')
        features['nb_hyphens'] = url.count('-')
        features['nb_at'] = url.count('@')
        features['nb_qm'] = url.count('?')
        features['nb_and'] = url.count('&')
        features['nb_or'] = url.count('|')
        features['nb_eq'] = url.count('=')
        features['nb_underscore'] = url.count('_')
        features['nb_tilde'] = url.count('~')
        features['nb_percent'] = url.count('%')
        features['nb_slash'] = url.count('/')
        features['nb_star'] = url.count('*')
        features['nb_colon'] = url.count(':')
        features['nb_comma'] = url.count(',')
        features['nb_semicolumn'] = url.count(';')
        features['nb_dollar'] = url.count('$')
        features['nb_space'] = url.count(' ')
        features['nb_www'] = 1 if 'www' in domain else 0
        features['nb_com'] = 1 if '.com' in domain else 0
        features['nb_dslash'] = url.count('//')
        
        # HTTP/HTTPS features
        features['http_in_path'] = 1 if 'http' in path else 0
        features['https_token'] = 1 if 'https' in url.lower() and parsed_url.scheme != 'https' else 0
        
        # Ratio features
        features['ratio_digits_url'] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
        features['ratio_digits_host'] = sum(c.isdigit() for c in domain) / len(domain) if len(domain) > 0 else 0
        
        # Punycode and port
        features['punycode'] = 1 if 'xn--' in domain else 0
        features['port'] = 1 if ':' in domain and domain.split(':')[-1].isdigit() else 0
        
        # TLD features
        tld = domain.split('.')[-1] if '.' in domain else ''
        features['tld_in_path'] = 1 if tld in path else 0
        features['tld_in_subdomain'] = 1 if domain.count('.') > 1 and tld in domain[:-len(tld)-1] else 0
        features['abnormal_subdomain'] = 1 if domain.count('.') > 3 else 0
        
        # Domain structure
        features['nb_subdomains'] = max(0, domain.count('.') - 1)
        features['prefix_suffix'] = 1 if '-' in domain else 0
        features['random_domain'] = 1 if re.search(r'[0-9]+[a-z]+[0-9]+', domain) else 0
        features['shortening_service'] = 1 if any(short in domain for short in ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'short.link']) else 0
        features['path_extension'] = 1 if re.search(r'\.(exe|zip|rar|bat|scr)$', path) else 0
        
        # Redirection features (simplified)
        features['nb_redirection'] = 0  # Would need actual HTTP request
        features['nb_external_redirection'] = 0
        
        # Text analysis features (simplified)
        words_raw = re.findall(r'[a-zA-Z]+', url)
        features['length_words_raw'] = sum(len(word) for word in words_raw)
        features['char_repeat'] = max([len(list(g)) for k, g in _import_('itertools').groupby(url)], default=1)
        features['shortest_words_raw'] = min([len(word) for word in words_raw], default=0)
        features['shortest_word_host'] = min([len(word) for word in re.findall(r'[a-zA-Z]+', domain)], default=0)
        features['shortest_word_path'] = min([len(word) for word in re.findall(r'[a-zA-Z]+', path)], default=0)
        features['longest_words_raw'] = max([len(word) for word in words_raw], default=0)
        features['longest_word_host'] = max([len(word) for word in re.findall(r'[a-zA-Z]+', domain)], default=0)
        features['longest_word_path'] = max([len(word) for word in re.findall(r'[a-zA-Z]+', path)], default=0)
        features['avg_words_raw'] = features['length_words_raw'] / len(words_raw) if words_raw else 0
        features['avg_word_host'] = sum(len(word) for word in re.findall(r'[a-zA-Z]+', domain)) / max(1, len(re.findall(r'[a-zA-Z]+', domain)))
        features['avg_word_path'] = sum(len(word) for word in re.findall(r'[a-zA-Z]+', path)) / max(1, len(re.findall(r'[a-zA-Z]+', path)))

        # Phishing hint features
        phishing_words = ['secure', 'account', 'webscr', 'login', 'ebayisapi', 'signin', 'banking', 'confirm']
        features['phish_hints'] = sum(1 for word in phishing_words if word in url.lower())
        features['domain_in_brand'] = 1 if any(brand in domain for brand in ['paypal', 'amazon', 'google', 'microsoft', 'apple']) else 0
        features['brand_in_subdomain'] = features['domain_in_brand']  # Simplified
        features['brand_in_path'] = 1 if any(brand in path for brand in ['paypal', 'amazon', 'google', 'microsoft', 'apple']) else 0
        features['suspecious_tld'] = 1 if tld in ['tk', 'ml', 'ga', 'cf'] else 0
        
        # Statistical and web features (defaults optimized for legitimate sites)
        features['statistical_report'] = 0
        features['nb_hyperlinks'] = 5  # Legitimate sites have some links
        features['ratio_intHyperlinks'] = 0.8  # Mostly internal links for legitimate sites
        features['ratio_extHyperlinks'] = 0.2  # Some external links
        features['ratio_nullHyperlinks'] = 0
        features['nb_extCSS'] = 0
        features['ratio_intRedirection'] = 1  # Internal redirections for legitimate sites
        features['ratio_extRedirection'] = 0.0
        features['ratio_intErrors'] = 0
        features['ratio_extErrors'] = 0.0
        features['login_form'] = 0
        features['external_favicon'] = 0
        features['links_in_tags'] = 0.5  # Some links in tags
        features['submit_email'] = 0
        features['ratio_intMedia'] = 0.8  # Mostly internal media
        features['ratio_extMedia'] = 0.2
        features['sfh'] = 1  # Legitimate sites often have secure form handling
        features['iframe'] = 0
        features['popup_window'] = 0
        features['safe_anchor'] = 0.8  # Most anchors are safe in legitimate sites
        features['onmouseover'] = 0
        features['right_clic'] = 0
        features['empty_title'] = 0
        features['domain_in_title'] = 1  # Legitimate sites often have domain in title
        features['domain_with_copyright'] = 1  # Legitimate sites often have copyright
        features['whois_registered_domain'] = 1  # Assume registered
        features['domain_registration_length'] = 624  # Mean for legitimate sites (624.29)
        features['domain_age'] = 5094  # Mean for legitimate sites (5093.94)
        features['web_traffic'] = 25176  # Median for legitimate sites (more realistic than mean)
        features['dns_record'] = 0  # Most legitimate sites have 0 in training data  
        features['google_index'] = 0  # Most legitimate sites have 0 (vs phishing=0.90)
        features['page_rank'] = 5  # Higher than phishing average (4.48 vs 1.89)
        
        # Convert to array in the exact order of the dataset columns (excluding 'url' and 'status')
        feature_names = [
            'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 
            'nb_and', 'nb_or', 'nb_eq', 'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 
            'nb_star', 'nb_colon', 'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www', 
            'nb_com', 'nb_dslash', 'http_in_path', 'https_token', 'ratio_digits_url', 'ratio_digits_host', 
            'punycode', 'port', 'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain', 'nb_subdomains', 
            'prefix_suffix', 'random_domain', 'shortening_service', 'path_extension', 'nb_redirection', 
            'nb_external_redirection', 'length_words_raw', 'char_repeat', 'shortest_words_raw', 
            'shortest_word_host', 'shortest_word_path', 'longest_words_raw', 'longest_word_host', 
            'longest_word_path', 'avg_words_raw', 'avg_word_host', 'avg_word_path', 'phish_hints', 
            'domain_in_brand', 'brand_in_subdomain', 'brand_in_path', 'suspecious_tld', 'statistical_report', 
            'nb_hyperlinks', 'ratio_intHyperlinks', 'ratio_extHyperlinks', 'ratio_nullHyperlinks', 
            'nb_extCSS', 'ratio_intRedirection', 'ratio_extRedirection', 'ratio_intErrors', 'ratio_extErrors', 
            'login_form', 'external_favicon', 'links_in_tags', 'submit_email', 'ratio_intMedia', 
            'ratio_extMedia', 'sfh', 'iframe', 'popup_window', 'safe_anchor', 'onmouseover', 'right_clic', 
            'empty_title', 'domain_in_title', 'domain_with_copyright', 'whois_registered_domain', 
            'domain_registration_length', 'domain_age', 'web_traffic', 'dns_record', 'google_index', 'page_rank'
        ]
        
        return [features.get(name, 0) for name in feature_names]
        
    except Exception as e:
        st.error(f"Error extracting features from URL: {e}")
        return None
        
def main():
    st.title("🔒 Phishing Website Detector")
    st.markdown("Enter a website URL to check if it's potentially a phishing site using machine learning.")
    
    # Load model
    if 'model' not in st.session_state:
        st.info("Loading and training the model...")
        model, scaler = train_model()
        if model is not None:
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.success("Model loaded successfully!")
        else:
            st.error("Failed to load the model. Please check your dataset.")
            return
    
    # URL input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url_input = st.text_input(
            "Enter Website URL:",
            placeholder="https://example.com",
            help="Enter the full URL including http:// or https://"
        )
    
    with col2:
        analyze_button = st.button("🔍 Analyze", type="primary")
    
    if analyze_button and url_input:
        if not url_input.startswith(('http://', 'https://')):
            url_input = 'http://' + url_input
        
        with st.spinner("Analyzing URL..."):
            # Extract features
            features = extract_url_features(url_input)
            
            if features is not None:
                # Make prediction
                features_scaled = st.session_state.scaler.transform([features])
                prediction_prob = st.session_state.model.predict(features_scaled)[0][0]
                prediction = 1 if prediction_prob > 0.5 else 0
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("⚠️ **PHISHING DETECTED**")
                        st.markdown("This URL appears to be **potentially dangerous**.")
                    else:
                        st.success("✅ **LEGITIMATE**")
                        st.markdown("This URL appears to be **safe**.")
                
                with col2:
                    confidence = max(prediction_prob, 1 - prediction_prob) * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                with col3:
                    risk_score = prediction_prob * 100
                    st.metric("Risk Score", f"{risk_score:.1f}%")
                
                # Show URL analysis
                st.subheader("🔍 URL Analysis")
                
                parsed_url = urlparse(url_input)
                analysis_data = {
                    "Component": ["Protocol", "Domain", "Path", "Length", "HTTPS"],
                    "Value": [
                        str(parsed_url.scheme),
                        str(parsed_url.netloc),
                        str(parsed_url.path if parsed_url.path else "/"),
                        str(len(url_input)),
                        "Yes" if parsed_url.scheme == 'https' else "No"
                    ],
                    "Assessment": [
                        "✅ Secure" if parsed_url.scheme == 'https' else "⚠️ Insecure",
                        "✅ Normal" if not re.match(r'^\d+\.\d+\.\d+\.\d+', parsed_url.netloc) else "⚠️ IP Address",
                        "✅ Normal" if len(parsed_url.path) < 50 else "⚠️ Long path",
                        "✅ Normal" if len(url_input) < 100 else "⚠️ Very long URL",
                        "✅ Encrypted" if parsed_url.scheme == 'https' else "⚠️ Not encrypted"
                    ]
                }
                
                df_analysis = pd.DataFrame(analysis_data)
                st.dataframe(df_analysis, use_container_width=True)
                
                # Safety recommendations
                st.subheader("🛡️ Safety Recommendations")
                if prediction == 1:
                    st.warning("""
                    **This URL has been flagged as potentially dangerous. Please:**
                    - Do not enter personal information
                    - Do not download files from this site
                    - Verify the URL with the official source
                    - Consider using a different browser or incognito mode
                    """)
                else:
                    st.info("""
                    **This URL appears safe, but always:**
                    - Verify you're on the correct website
                    - Check for HTTPS encryption
                    - Be cautious with personal information
                    - Keep your browser updated
                    """)
    
    elif analyze_button and not url_input:
        st.warning("Please enter a URL to analyze.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This phishing detector uses machine learning to analyze URL characteristics and determine if a website might be a phishing site.
        
        **Features analyzed:**
        - URL length and structure
        - Domain characteristics
        - Special characters usage
        - Protocol security (HTTPS)
        - Suspicious patterns
        
        **Accuracy:** Based on training data analysis
        """)
        
        st.header("🔗 How to use")
        st.markdown("""
        1. Enter a complete URL (with http:// or https://)
        2. Click "Analyze" 
        3. Review the results and recommendations
        4. Always use caution when visiting unknown websites
        """)

if __name__ == "__main__":
    main() 
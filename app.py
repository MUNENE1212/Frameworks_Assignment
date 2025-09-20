# Solution 1: Increase Streamlit upload limit
# Create a config.toml file in .streamlit folder

# .streamlit/config.toml
maxUploadSize = 2000  # Size in MB (default is 200MB)

# Solution 2: Modified Streamlit app with local file reading
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import re
from collections import Counter
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="CORD-19 Analysis Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_local_data(file_path, sample_size=None):
    """Load data from local file with optional sampling"""
    try:
        if sample_size:
            # Read in chunks and sample randomly
            chunk_list = []
            chunk_size = 10000
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_rows = sum(1 for line in open(file_path)) - 1  # Subtract header
            chunks_needed = min(total_rows // chunk_size, sample_size // chunk_size + 1)
            
            for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                if i >= chunks_needed:
                    break
                    
                chunk_list.append(chunk)
                progress = (i + 1) / chunks_needed
                progress_bar.progress(progress)
                status_text.text(f'Loading data... {i+1}/{chunks_needed} chunks')
            
            df = pd.concat(chunk_list, ignore_index=True)
            
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                
            progress_bar.empty()
            status_text.empty()
            
        else:
            # Load entire file with progress tracking
            with st.spinner('Loading full dataset... This may take a few minutes.'):
                df = pd.read_csv(file_path)
        
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_uploaded_data(uploaded_file):
    """Load and cache uploaded data"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading uploaded data: {e}")
        return None

@st.cache_data
def clean_data(df):
    """Clean and prepare the dataset"""
    df_clean = df.copy()
    
    # Remove rows without title
    if 'title' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['title'])
    
    # Handle publish_time conversion
    if 'publish_time' in df_clean.columns:
        df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
        df_clean['publication_year'] = df_clean['publish_time'].dt.year
        
        # Filter for reasonable years
        valid_years = (df_clean['publication_year'] >= 1990) & (df_clean['publication_year'] <= 2024)
        df_clean = df_clean[valid_years | df_clean['publication_year'].isnull()]
    
    # Create derived columns
    if 'abstract' in df_clean.columns:
        df_clean['abstract_word_count'] = df_clean['abstract'].astype(str).apply(
            lambda x: len(x.split()) if pd.notna(x) and x != 'nan' else 0
        )
    
    if 'title' in df_clean.columns:
        df_clean['title_length'] = df_clean['title'].astype(str).apply(len)
    
    if 'journal' in df_clean.columns:
        df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
    
    return df_clean

def create_data_sampler():
    """Create interface for data sampling"""
    st.sidebar.subheader("📊 Data Loading Options")
    
    # File source selection
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload file", "Local file"]
    )
    
    # Sampling options
    use_sampling = st.sidebar.checkbox("Use data sampling (recommended for large files)", value=True)
    
    if use_sampling:
        sample_size = st.sidebar.slider(
            "Sample size (number of papers)",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
            help="Smaller samples load faster but may miss patterns in the data"
        )
    else:
        sample_size = None
        st.sidebar.warning("⚠️ Loading full dataset may take several minutes and use significant memory")
    
    return data_source, sample_size

# All the visualization functions from the previous version
def create_yearly_plot(df, year_range):
    """Create publication by year plot"""
    if 'publication_year' not in df.columns:
        return None
    
    df_filtered = df[(df['publication_year'] >= year_range[0]) & 
                     (df['publication_year'] <= year_range[1])]
    
    yearly_counts = df_filtered['publication_year'].value_counts().sort_index()
    
    fig = px.bar(x=yearly_counts.index, y=yearly_counts.values,
                 labels={'x': 'Publication Year', 'y': 'Number of Papers'},
                 title='COVID-19 Research Publications by Year')
    fig.update_layout(height=500)
    return fig

def create_journal_plot(df, top_n):
    """Create top journals plot"""
    if 'journal' not in df.columns:
        return None
    
    journal_counts = df['journal'].value_counts().head(top_n)
    
    fig = px.bar(x=journal_counts.values, y=journal_counts.index,
                 orientation='h',
                 labels={'x': 'Number of Papers', 'y': 'Journal'},
                 title=f'Top {top_n} Journals Publishing COVID-19 Research')
    fig.update_layout(height=max(400, top_n * 30))
    return fig

def create_wordcloud(df):
    """Create word cloud from titles"""
    if 'title' not in df.columns:
        return None
    
    try:
        all_titles = ' '.join(df['title'].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             max_words=100).generate(all_titles)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating word cloud: {e}")
        return None

def analyze_word_frequency(df, top_n):
    """Analyze title word frequency"""
    if 'title' not in df.columns:
        return None
    
    all_titles = ' '.join(df['title'].dropna().astype(str))
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
    
    stop_words = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 
                  'they', 'been', 'have', 'has', 'had', 'was', 'were', 'will', 
                  'would', 'could', 'should', 'can', 'may', 'might', 'must'}
    
    filtered_words = [word for word in words if word not in stop_words]
    word_freq = Counter(filtered_words)
    top_words = word_freq.most_common(top_n)
    
    words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    
    fig = px.bar(words_df, x='Word', y='Frequency',
                 title=f'Top {top_n} Most Frequent Words in Titles')
    fig.update_layout(height=500)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🦠 CORD-19 Research Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard analyzes the CORD-19 dataset metadata to provide insights into COVID-19 research publications.
    For large files, use sampling or place the metadata.csv file in the same directory as this script.
    """)
    
    # Data source selection
    data_source, sample_size = create_data_sampler()
    
    df = None
    
    if data_source == "Upload file":
        # File upload with increased limit message
        uploaded_file = st.sidebar.file_uploader(
            "Upload CORD-19 metadata.csv",
            type=['csv'],
            help="Note: Default upload limit is 200MB. For larger files, use 'Local file' option."
        )
        
        if uploaded_file is not None:
            with st.spinner('Loading uploaded data...'):
                df = load_uploaded_data(uploaded_file)
                if df is not None and sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
    
    else:  # Local file
        # Check for metadata.csv in current directory
        local_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        
        if 'metadata.csv' in local_files:
            if st.sidebar.button("Load metadata.csv"):
                df = load_local_data('metadata.csv', sample_size)
        else:
            st.sidebar.error("metadata.csv not found in current directory")
            
            # Show available CSV files
            if local_files:
                selected_file = st.sidebar.selectbox(
                    "Or select another CSV file:",
                    [''] + local_files
                )
                if selected_file and st.sidebar.button(f"Load {selected_file}"):
                    df = load_local_data(selected_file, sample_size)
    
    if df is not None:
        # Clean data
        with st.spinner('Cleaning data...'):
            df_clean = clean_data(df)
        
        st.success(f"Data loaded successfully! {len(df_clean):,} papers after cleaning.")
        
        if sample_size and len(df_clean) == sample_size:
            st.info(f"📊 Showing analysis of {sample_size:,} randomly sampled papers from the dataset.")
        
        # Sidebar controls
        st.sidebar.subheader("📈 Analysis Controls")
        
        # Year range selector
        if 'publication_year' in df_clean.columns:
            min_year = int(df_clean['publication_year'].min())
            max_year = int(df_clean['publication_year'].max())
            year_range = st.sidebar.slider(
                "Publication Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
        else:
            year_range = (2000, 2024)
        
        # Top N selectors
        top_journals_n = st.sidebar.slider("Number of Top Journals", 5, 30, 15)
        top_words_n = st.sidebar.slider("Number of Top Words", 10, 50, 20)
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "📰 Journals", "🔤 Text Analysis"])
        
        with tab1:
            st.header("Dataset Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Papers", f"{len(df_clean):,}")
            
            with col2:
                if 'publication_year' in df_clean.columns:
                    year_span = df_clean['publication_year'].max() - df_clean['publication_year'].min()
                    st.metric("Year Span", f"{year_span:.0f} years")
            
            with col3:
                if 'journal' in df_clean.columns:
                    unique_journals = df_clean['journal'].nunique()
                    st.metric("Unique Journals", f"{unique_journals:,}")
            
            with col4:
                if 'abstract_word_count' in df_clean.columns:
                    avg_abstract = df_clean['abstract_word_count'].mean()
                    st.metric("Avg Abstract Length", f"{avg_abstract:.0f} words")
            
            # Data sample
            st.subheader("Data Sample")
            display_columns = ['title', 'authors', 'journal', 'publish_time', 'abstract']
            available_columns = [col for col in display_columns if col in df_clean.columns]
            st.dataframe(df_clean[available_columns].head(10), use_container_width=True)
            
            # Missing data analysis
            st.subheader("Data Quality")
            missing_data = df_clean.isnull().sum().sort_values(ascending=False)
            missing_pct = (missing_data / len(df_clean)) * 100
            
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': missing_pct.values
            }).head(10)
            
            fig_missing = px.bar(missing_df, x='Column', y='Missing %',
                               title='Missing Data by Column (Top 10)')
            st.plotly_chart(fig_missing, use_container_width=True)
        
        with tab2:
            st.header("Publication Trends")
            
            # Publications by year
            if 'publication_year' in df_clean.columns:
                fig_yearly = create_yearly_plot(df_clean, year_range)
                if fig_yearly:
                    st.plotly_chart(fig_yearly, use_container_width=True)
                
                # Year-over-year analysis
                df_filtered = df_clean[(df_clean['publication_year'] >= year_range[0]) & 
                                     (df_clean['publication_year'] <= year_range[1])]
                yearly_stats = df_filtered.groupby('publication_year').size()
                
                col1, col2 = st.columns(2)
                with col1:
                    peak_year = yearly_stats.idxmax()
                    st.metric("Peak Publication Year", f"{peak_year:.0f}", 
                            f"{yearly_stats.max():,} papers")
                
                with col2:
                    if len(yearly_stats) > 1:
                        growth_rate = ((yearly_stats.iloc[-1] - yearly_stats.iloc[-2]) / 
                                     yearly_stats.iloc[-2]) * 100
                        st.metric("Year-over-Year Growth", f"{growth_rate:+.1f}%")
            else:
                st.warning("No publication year data available")
        
        with tab3:
            st.header("Journal Analysis")
            
            if 'journal' in df_clean.columns:
                # Top journals plot
                fig_journals = create_journal_plot(df_clean, top_journals_n)
                if fig_journals:
                    st.plotly_chart(fig_journals, use_container_width=True)
                
                # Journal statistics
                journal_stats = df_clean['journal'].value_counts()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Journals", f"{len(journal_stats):,}")
                
                with col2:
                    top_journal_pct = (journal_stats.iloc[0] / len(df_clean)) * 100
                    st.metric("Top Journal Share", f"{top_journal_pct:.1f}%")
                
                with col3:
                    # Calculate concentration (top 10 journals share)
                    top_10_share = (journal_stats.head(10).sum() / len(df_clean)) * 100
                    st.metric("Top 10 Journals Share", f"{top_10_share:.1f}%")
                
                # Detailed journal table
                st.subheader("Detailed Journal Statistics")
                journal_df = pd.DataFrame({
                    'Journal': journal_stats.head(20).index,
                    'Paper Count': journal_stats.head(20).values,
                    'Percentage': (journal_stats.head(20).values / len(df_clean) * 100).round(2)
                })
                st.dataframe(journal_df, use_container_width=True)
            else:
                st.warning("No journal data available")
        
        with tab4:
            st.header("Text Analysis")
            
            # Word frequency analysis
            fig_words = analyze_word_frequency(df_clean, top_words_n)
            if fig_words:
                st.plotly_chart(fig_words, use_container_width=True)
            
            # Word cloud
            st.subheader("Title Word Cloud")
            fig_wordcloud = create_wordcloud(df_clean)
            if fig_wordcloud:
                st.pyplot(fig_wordcloud)
            else:
                st.info("Install wordcloud package to see word cloud: pip install wordcloud")
            
            # Text statistics
            if 'abstract_word_count' in df_clean.columns:
                st.subheader("Abstract Length Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    avg_length = df_clean['abstract_word_count'].mean()
                    st.metric("Average Abstract Length", f"{avg_length:.0f} words")
                
                with col2:
                    median_length = df_clean['abstract_word_count'].median()
                    st.metric("Median Abstract Length", f"{median_length:.0f} words")
                
                # Abstract length distribution
                fig_abstract = px.histogram(df_clean, x='abstract_word_count',
                                          title='Distribution of Abstract Lengths',
                                          labels={'abstract_word_count': 'Word Count', 'count': 'Frequency'})
                fig_abstract.update_layout(height=400)
                st.plotly_chart(fig_abstract, use_container_width=True)
            
            # Title length analysis
            if 'title_length' in df_clean.columns:
                st.subheader("Title Length Analysis")
                
                fig_title_len = px.box(df_clean, y='title_length',
                                     title='Distribution of Title Lengths (Characters)')
                fig_title_len.update_layout(height=400)
                st.plotly_chart(fig_title_len, use_container_width=True)
        
        # Download section
        st.sidebar.subheader("💾 Export Data")
        
        # Prepare export data
        export_data = df_clean.copy()
        csv = export_data.to_csv(index=False)
        
        st.sidebar.download_button(
            label="Download Cleaned Data as CSV",
            data=csv,
            file_name="cord19_cleaned_data.csv",
            mime="text/csv"
        )
        
        # Summary statistics
        st.sidebar.subheader("📊 Quick Stats")
        st.sidebar.write(f"**Total Papers:** {len(df_clean):,}")
        
        if 'publication_year' in df_clean.columns:
            st.sidebar.write(f"**Year Range:** {df_clean['publication_year'].min():.0f} - {df_clean['publication_year'].max():.0f}")
        
        if 'journal' in df_clean.columns:
            st.sidebar.write(f"**Unique Journals:** {df_clean['journal'].nunique():,}")
        
        if 'abstract_word_count' in df_clean.columns:
            avg_abstract = df_clean['abstract_word_count'].mean()
            st.sidebar.write(f"**Avg Abstract:** {avg_abstract:.0f} words")

    else:
        # Instructions when no file is loaded
        st.info("👆 Please upload a file or place metadata.csv in the current directory")
        
        st.markdown("""
        ### Large File Solutions:
        
        **Option 1: Increase Upload Limit**
        Create `.streamlit/config.toml` file:
        ```toml
        [server]
        maxUploadSize = 1000  # Size in MB
        ```
        
        **Option 2: Use Local File (Recommended)**
        1. Place `metadata.csv` in the same directory as this script
        2. Select "Local file" option in the sidebar
        3. Use sampling for faster loading
        
        **Option 3: Pre-process the Data**
        ```python
        # Create a smaller sample file
        import pandas as pd
        df = pd.read_csv('metadata.csv')
        df_sample = df.sample(n=50000)
        df_sample.to_csv('metadata_sample.csv', index=False)
        ```
        
        ### Memory Management Tips:
        - Use data sampling for exploratory analysis
        - Close other applications to free memory
        - Consider chunked processing for full dataset analysis
        """)

if __name__ == "__main__":
    main()
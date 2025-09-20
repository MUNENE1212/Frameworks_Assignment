# CORD-19 Research Analysis Project

A comprehensive analysis toolkit for exploring COVID-19 research publications using the CORD-19 dataset.

## Overview

This project provides tools to analyze patterns in COVID-19 research publications, including publication trends, journal distributions, text mining, and content analysis. It's designed to handle large datasets efficiently through sampling and provides both Jupyter notebook analysis and interactive Streamlit dashboard interfaces.

## Features

- **Data Loading & Sampling**: Efficiently handle large CORD-19 files with intelligent sampling
- **Publication Trend Analysis**: Track research output over time
- **Journal Analysis**: Identify top publishing venues and concentration patterns
- **Text Mining**: Word frequency analysis and content themes
- **Interactive Visualizations**: Charts and plots using matplotlib, seaborn, and plotly
- **Data Export**: Save cleaned datasets and analysis results
- **Multiple Interfaces**: Jupyter notebook for analysis, Streamlit for interactive exploration

## Project Structure

```
cord19-analysis/
│
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── cord19_analysis.py           # Standalone analysis script
├── streamlit_app.py             # Interactive dashboard
├── cord_19.ipynb        # Jupyter notebook (main analysis)
                 
```

## Installation

### 1. Clone or Download
```bash
git clone [repository-url]
cd cord19-analysis
```

### 2. Create Virtual Environment
```bash
python -m venv cord19_env
source cord19_env/bin/activate  # On Windows: cord19_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download CORD-19 Data
1. Visit the [CORD-19 download page](https://www.semanticscholar.org/cord19/download)
2. Download `metadata.csv` file (approximately 500MB)
3. Place it in the `data/` directory or project root

## Usage

### Option 1: Jupyter Notebook (Recommended for Analysis)

```bash
jupyter lab cord19_analysis.ipynb
```

The notebook is organized into four main parts:
1. **Data Loading & Exploration** (2-3 hours)
2. **Data Cleaning & Preparation** (2-3 hours)
3. **Analysis & Visualization** (3-4 hours)
4. **Summary & Export** (1 hour)

### Option 2: Streamlit Dashboard (Interactive Exploration)

```bash
streamlit run streamlit_app.py
```

Features:
- File upload interface (handles large files with configuration)
- Interactive filtering and parameter adjustment
- Multiple analysis tabs (Overview, Trends, Journals, Text Analysis)
- Real-time visualizations
- Data export functionality

### Option 3: Python Script (Automated Analysis)

```bash
python cord19_analysis.py
```

## Key Functions

### Data Loading
```python
# Load full dataset
df = load_cord19_data("metadata.csv")

# Load sample with auto-save
df = load_cord19_data("metadata.csv", sample_size=10000, save_sample=True)
```

### Analysis Functions
- `analyze_publication_trends()`: Temporal analysis of research output
- `analyze_journals()`: Publishing venue analysis
- `analyze_text_content()`: Word frequency and text mining
- `clean_cord19_data()`: Data cleaning and preparation

## Configuration

### Large File Handling
For files larger than 200MB, create `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 1000  # Size in MB
maxMessageSize = 1000
```

### Memory Optimization
- Use sampling for initial exploration: `sample_size=10000`
- Close unused applications when processing large datasets
- Consider chunked processing for full dataset analysis

## Sample Outputs

### Publication Trends
- Year-over-year publication patterns
- COVID-19 era research surge analysis
- Peak publication periods

### Journal Analysis
- Top publishing venues
- Publication concentration metrics
- Journal distribution patterns

### Text Analysis
- Most frequent terms in titles
- Word clouds and content themes
- Abstract length distributions

## Data Quality Features

- Missing data assessment and visualization
- Data completeness metrics
- Logical cleaning for research metadata
- Publication date validation
- Author and journal standardization

## Export Options

- Cleaned datasets (CSV format)
- Analysis summaries (text reports)
- Sample datasets for sharing
- Visualization plots (PNG/SVG)

## Common Issues & Solutions

### Memory Errors
- Reduce sample size: `sample_size=5000`
- Use chunked loading for large files
- Restart kernel between analyses

### File Not Found
- Verify `metadata.csv` is in correct directory
- Check file path in loading function
- Ensure file is fully downloaded (500MB+)

### Missing Dependencies
```bash
pip install pandas numpy matplotlib seaborn plotly wordcloud streamlit
```

### Slow Performance
- Use smaller samples for testing
- Enable sampling in Streamlit interface
- Consider running analysis on a subset of columns

## Requirements

- Python 3.8+
- 4GB+ RAM (8GB recommended for full dataset)
- 2GB free disk space
- Internet connection for initial data download

## Dependencies

See `requirements.txt` for complete list. Key packages:
- pandas >= 1.5.0
- numpy >= 1.24.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- plotly >= 5.15.0
- streamlit >= 1.28.0
- wordcloud >= 1.9.0

## Citation

If you use this analysis in research or publications, please cite:

1. The CORD-19 dataset: [https://www.semanticscholar.org/cord19](https://www.semanticscholar.org/cord19)
2. This analysis framework (provide your repository/publication details)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request

## License

[Specify your license - MIT, Apache 2.0, etc.]

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the Jupyter notebook documentation
3. Create an issue in the repository

## Roadmap

Future enhancements:
- Author network analysis
- Geographic research distribution
- Topic modeling (LDA/BERT)
- Citation network analysis
- Real-time dataset updates
- API development for automated analysis

---

**Note**: This project is for research and educational purposes. Follow CORD-19 dataset terms of use and cite appropriately in any publications.
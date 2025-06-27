import os
import sys
import subprocess
import pandas as pd
from together import Together
from typing import Optional, List, Dict, Any, Tuple, Callable
import tiktoken
import re
import streamlit as st
import tempfile
import io
import zipfile
import base64
import shutil
import seaborn as sns

# Detect Environment
def detect_environment():
    # Streamlit Cloud sets this in hosted mode
    if os.environ.get('STREAMLIT_SERVER_HEADLESS', '') == '1':
        return 'cloud'
    elif os.path.exists('/.dockerenv'):
        return 'docker'
    else:
        return 'local'

# Cleanup generated files
def cleanup_generated_files():
    files_to_clean = ['generated_code.py', 'output_plot.png', 'transformed_data.csv', 'transformed_data.xlsx', 'transformed_data.json']
    for file in files_to_clean:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                st.warning(f"Could not remove {file}: {str(e)}")

# Forbidden Code Check
def contains_forbidden_code(code: str) -> bool:
    forbidden_terms = [
        'eval', 'exec ', 'exec(', 'os.system', 'os.remove', 'os.rmdir', 'os.unlink',
        'import os', 'shutil', '__import__', 'open(', 'socket',
        'input(', 'exit(', 'quit(', 'del '
    ]
    for term in forbidden_terms:
        if term in code:
            print(f"Forbidden term found: {term}")
            return True
    # Allow 'import sys' and 'subprocess' only for package install
    if ('import sys' in code or 'subprocess' in code) and "subprocess.check_call([sys.executable, '-m', 'pip'" not in code:
        print("package install problem")
        return True
    return False

# Prompt Intent Check
def check_prompt_intent(prompt: str, call_llm) -> bool:
    intent_check_prompt = f"""
    Evaluate the following prompt for use with a data visualization agent.
    Return only 'valid' or 'invalid'. It must only describe a plot, chart,
    table, or data manipulation operation on a dataset. No file access, 
    no system commands, no unrelated instructions.

    Prompt: "{prompt}"
    """
    response = call_llm(intent_check_prompt)
    return 'valid' in response.lower()

# Page Config
st.set_page_config(
    page_title="Dataset Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initiate the Session State
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'dataset_dict' not in st.session_state:
    st.session_state.dataset_dict = {'Example: penguins': sns.load_dataset('penguins')}
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = 'Example: penguins'
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = None
if 'execution_output' not in st.session_state:
    st.session_state.execution_output = None
if 'generated_files' not in st.session_state:
    st.session_state.generated_files = []

# Defines Agent Class
class DatasetAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.supported_formats = {
            '.csv': ('CSV', pd.read_csv),
            '.tsv': ('TSV', lambda x: pd.read_csv(x, sep='\t')),
            '.txt': ('TSV', lambda x: pd.read_csv(x, sep='\t')),
            '.xlsx': ('XLSX', pd.read_excel),
            '.xls': ('XLS', pd.read_excel),
            '.parquet': ('PARQUET', pd.read_parquet),
            '.json': ('JSON', pd.read_json),
            '.feather': ('FEATHER', pd.read_feather)
        }
        self.current_file = None
        self.default_output_names = {
            'plot': 'output_plot.png',
            'data': 'transformed_data.csv',
            'excel': 'transformed_data.xlsx',
            'json': 'transformed_data.json'
        }

    def validate_file_path(self, file_path: str) -> Tuple[bool, str]:
        # Validate if the file exists and is in a supported format
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_formats:
            return False, f"Unsupported file format: {ext}. Supported formats: {', '.join(self.supported_formats.keys())}"
        return True, ""

    # Return the file format and appropriate reader function
    def check_file_extension(self, file_path: str) -> Tuple[str, Callable]:
        ext = os.path.splitext(file_path)[1].lower()
        return self.supported_formats.get(ext, ("Unknown", None))

    # Validate generated code if exists and is correct format
    def validate_generated_code(self, code: str) -> Tuple[bool, str]:
        if not code or not isinstance(code, str):
            return False, "No code was generated or invalid code type"
        # Strip any potential markdown formatting
        if code.startswith("```python"):
            code = code.split("```python")[1]
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
        code = code.strip()
        # Regex check for risk terms in code call function contains_forbidden_code above
        if contains_forbidden_code(code):
            return False, "Code contains forbidden or unsafe terms."
        required_terms = ['import pandas', 'importlib.util']
        for term in required_terms:
            if term not in code:
                return False, f"Code is missing required term: {term}"
        # Check for basic syntax errors
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            st.error(f"\nDebug - Raw code received:")
            st.code(code)
            st.error(f"\nSyntax error details: {str(e)}")
            return False, f"Code contains syntax error: {str(e)}"
        return True, code

    def get_analysis_prompt(self, datasetColumns: list, datasetRows: list, datasetTypes: list, query: str) -> str:
        return f"""Given a dataset with the following structure, write Python code to {query}

Dataset Information:
- Columns: {datasetColumns}
- Sample Data (first 5 rows): {datasetRows[:5]}
- Column Types: {datasetTypes}
- Total Rows: {len(datasetRows)}

Reassess the query 3 times to ensure all aspects of the query are accounted for in the code solution.
Assess 3 different code solutions for performing the query request and 1. select the option that best fits the query and 2. is the simplest

Requirements:
1. Start with a package management section that:
   - Lists all required packages in a list variable
   - Checks if each package is installed using importlib.util.find_spec
   - Installs missing packages using pip (subprocess.check_call)
   - Only then imports the required packages
2. Use pandas for data manipulation, if a prompt request needs data preprocessing such as dropping missing values (dropna()) preform these operations.
3. Read the dataset from '{self.current_file}'
4. For any outputs:
   - Plots/graphs save to '{self.default_output_names['plot']}'
   - save figures with high resolution and without empty space (dpi=300, bbox_inches='tight')
   - Transformed data save to '{self.default_output_names['data']}'
5. Include error handling, assume the dataset always exists and do not use exit().
6. Add comments explaining each section
7. Use full row indices for any row-specific operations

Example package management format:
```python
import importlib.util
import subprocess
import sys

# List of required packages
required_packages = ['pandas', 'matplotlib', 'seaborn']  # Add any packages you need

# Check and install missing packages
for package in required_packages:
    if importlib.util.find_spec(package) is None:
        print(f"{'package'} not found. Installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'package'])
    else:
        print(f"{'package'} is already installed.")

# Now import the required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

Return only the Python code without any additional text or markdown."""

    def get_code(self, file_path: str, query: str):
        try:
            # Validate file path, call validate_file_path function above
            is_valid, error_message = self.validate_file_path(file_path)
            if not is_valid:
                st.error(error_message)
                return None
            self.current_file = file_path
            # Get file type and read function, call check_file_extension function above
            file_format, reader_func = self.check_file_extension(file_path)
            if reader_func is None:
                return None
            df = reader_func(file_path)
            datasetColumns = df.columns.tolist()
            datasetRows = df.values.tolist()
            datasetTypes = df.dtypes.tolist()
            prompt = self.get_analysis_prompt(datasetColumns, datasetRows, datasetTypes, query)
            # Get code from LLM
            with st.spinner("Generating code with AI..."):
                code_analysis = self.llm.invoke(prompt)
            ## Debugging
            #print("RAW GENERATED CODE:")
            #print(code_analysis)
            #st.code(code_analysis, language='python')
            # Validate generated code
            is_valid, result = self.validate_generated_code(code_analysis)
            if not is_valid:
                st.error(f"Generated code validation failed: {result}")
                return None
            # Save the validated and cleaned code
            with open('generated_code.py', 'w') as file:
                file.write(result)
            st.success("Cleaned and validated code saved to generated_code.py")
            return result
        except Exception as e:
            st.error(f"Error in get_code: {str(e)}")
            return None

class TogetherEndpoint:
    def __init__(self, model_id: str, api_key: str, temperature: float = 0.2, max_tokens: int = 1500):
        os.environ['TOGETHER_API_KEY'] = api_key
        self.client = Together()
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def invoke(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error with Together API: {str(e)}")
            raise

# Create agent, called by main(), calls both classes above
def create_agents(api_key, temperature=0.2, max_tokens=1500):
    llm = TogetherEndpoint(
        model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    dataset_agent = DatasetAgent(llm)
    return dataset_agent

# Main function, calls all other functions above, creates the app layout
def main():
    st.markdown('<h1 class="main-header">Data Visualization and Transformation Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered data analysis and visualization tool. Enter your Together API key, select or upload a dataset, and describe your data analysis or visualization request. The agent will generate and execute code, displaying results and allowing you to download all outputs.</p>', unsafe_allow_html=True)

    # dashboard app API key input 
    st.subheader("Together API Key")
    api_key = st.text_input("Enter your Together API key", type="password", value=st.session_state.api_key, help="Your key is only stored in this session and never saved.")
    st.session_state.api_key = api_key
    if not api_key:
        st.info("Please enter your Together API key to use the app.")
        #st.stop()

    # Dataset selection, add any uploaded files to the dropdown, deafualt is penguins
    st.subheader("Select or Upload Dataset")
    dataset_options = list(st.session_state.dataset_dict.keys())
    # Add any uploaded files to the dropdown
    for fname in st.session_state.uploaded_files:
        if fname not in st.session_state.dataset_dict:
            st.session_state.dataset_dict[fname] = st.session_state.uploaded_files[fname]
    selected = st.selectbox("Choose dataset", dataset_options, index=dataset_options.index(st.session_state.selected_dataset) if st.session_state.selected_dataset in dataset_options else 0)
    st.session_state.selected_dataset = selected

    st.markdown("<br><br>", unsafe_allow_html=True)

    # assess uploaded files, if valid, add to the dropdown
    uploaded_file = st.file_uploader("Upload a dataset file (CSV, TSV, TXT, XLSX, XLS, PARQUET, JSON, FEATHER)", type=['csv', 'tsv', 'txt', 'xlsx', 'xls', 'parquet', 'json', 'feather'])
    if uploaded_file is not None:
        try:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext in ['.csv', '.tsv', '.txt']:
                sep = '\t' if ext in ['.tsv', '.txt'] else ','
                df = pd.read_csv(uploaded_file, sep=sep)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file)
            elif ext == '.parquet':
                df = pd.read_parquet(uploaded_file)
            elif ext == '.json':
                df = pd.read_json(uploaded_file)
            elif ext == '.feather':
                df = pd.read_feather(uploaded_file)
            else:
                st.error("Unsupported file format.")
                df = None
            if df is not None:
                st.session_state.uploaded_files[uploaded_file.name] = df
                st.session_state.dataset_dict[uploaded_file.name] = df
                st.session_state.selected_dataset = uploaded_file.name
                st.success(f"File uploaded and added: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading uploaded file: {str(e)}")

    # Display the selected dataset summary
    df = st.session_state.dataset_dict[st.session_state.selected_dataset]
    st.markdown(f"**Selected Dataset:** `{st.session_state.selected_dataset}`")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True)
    st.write("**Column Info:**")
    col_info = pd.DataFrame({
        'Column': df.columns,
        #'Type': df.dtypes,
        'Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum()
    })
    st.dataframe(col_info, use_container_width=True)

    # User prompt input for data visualization or transformation of selected dataset
    st.subheader("Describe Your Data Task")
    prompt = st.text_area(
        "Describe what you want to do (e.g., plot, transform, summarize)",
        placeholder="e.g., Create a bar chart of species count, or Filter rows where bill_length_mm > 40",
        height=120
    )
    with st.expander("Example Prompts"):
        st.write("""
        - "Create a bar chart showing species count"
        - "Filter the data to show only rows where bill_length_mm > 40"
        - "Create a scatter plot of flipper_length_mm vs body_mass_g"
        - "Calculate the mean and standard deviation of all numeric columns"
        - "Group the data by island and show the count"
        """)

    # Submit/send button
    if st.button("Submit", type="primary", use_container_width=True):
        # Clean up any previously generated files
        cleanup_generated_files()
        
        # Check the intent of the prompt 
        # Set up model for intent check
        def call_llm_for_intent(prompt):
            # Use Together API for intent check (short, low-temp, low-tokens)
            llm = TogetherEndpoint(
                model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                api_key=api_key,
                temperature=0.0,
                max_tokens=10
            )
            return llm.invoke(prompt)
        if not check_prompt_intent(prompt, call_llm_for_intent):
            st.warning("Your prompt appears to be unrelated to data exploration.")
            st.stop()
        # Save dataset to temp file for agent
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            dataset_path = tmp_file.name
        # Create agent
        agent = create_agents(api_key)
        # Generate code
        code = agent.get_code(dataset_path, prompt)
        if not code:
            st.error("Failed to generate code.")
            st.stop()
        st.session_state.generated_code = code
        # Detect environment
        env = detect_environment()
        # Run code securely
        output_files = []
        try:
            if env == 'cloud':
                # Streamlit Community Cloud run with subprocess.run()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as code_file:
                    code_file.write(code.encode('utf-8'))
                    code_file_path = code_file.name
                result = subprocess.run([sys.executable, code_file_path], capture_output=True, text=True, timeout=60)
                output = result.stdout + '\n' + result.stderr
            elif env == 'local':
                # Local run with subprocess.run(), potentially in Docker (placeholder below)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as code_file:
                    code_file.write(code.encode('utf-8'))
                    code_file_path = code_file.name
                # Docker sandbox placeholder
                # To enable, run: docker run --rm -v $(pwd):/app python:3.10 python /app/generated_code.py
                result = subprocess.run([sys.executable, code_file_path], capture_output=True, text=True, timeout=60)
                output = result.stdout + '\n' + result.stderr
            else:
                st.error("Unknown environment. Cannot run code.")
                output = ""
            st.session_state.execution_output = output
            # Check for output files
            files_to_check = ['generated_code.py', 'output_plot.png', 'transformed_data.csv']
            st.session_state.generated_files = []
            for fname in files_to_check:
                if os.path.exists(fname):
                    st.session_state.generated_files.append(fname)
        except Exception as e:
            st.session_state.execution_output = f"Error running generated code: {str(e)}"

    # Results section display
    if st.session_state.generated_code:
        # Show figure if exists
        if 'output_plot.png' in st.session_state.generated_files:
            st.image('output_plot.png', caption='Generated Figure', use_column_width=True)
        # Show transformed data preview if exists 
        if 'transformed_data.csv' in st.session_state.generated_files:
            try:
                df_out = pd.read_csv('transformed_data.csv')
                st.dataframe(df_out.head(10), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display transformed data: {str(e)}")
        # Show code output mainly for error reporting 
        if st.session_state.execution_output:
            st.text_area("Execution Output", st.session_state.execution_output, height=120)
        # Download all generated files button, zip and download all files
        if st.session_state.generated_files:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, 'results.zip')
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for fname in st.session_state.generated_files:
                        zipf.write(fname, arcname=os.path.basename(fname))
                with open(zip_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/zip;base64,{b64}" download="results.zip">Download All Results</a>'
                    st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
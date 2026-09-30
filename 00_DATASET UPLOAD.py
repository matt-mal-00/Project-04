# ==========================================
# STEP 0: DATASET UPLOAD (GOOGLE COLAB ONLY)
# ==========================================
from google.colab import files
import io
import pandas as pd

print("Please upload your web analytics dataset (CSV format):")
uploaded = files.upload()

# Get the exact filename dynamically
file_name = next(iter(uploaded))
print(f"File '{file_name}' successfully uploaded and ready for analysis!")
import sys
import pandas as pd

def get_dataset(html_filename):
    html_data = pd.read_html(html_filename)

    print(html_data)

html_filename = sys.argv[1]
get_dataset(html_filename)





# Read all tables from the HTML file
tables = pd.read_html("path/to/file.html")

# tables is a list of DataFrames
print(f"Found {len(tables)} tables")

# Select the table you want (e.g., the first one)
df = tables[0]

# Preview the dataset
print(df.head())

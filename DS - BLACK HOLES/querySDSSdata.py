import requests
import pandas as pd

# Define the SQL query
query = """
SELECT
    objid,
    ra,
    dec,
    petroR90_r AS size,
    modelMag_r AS brightness
FROM
    Galaxy
WHERE
    petroR90_r IS NOT NULL
    AND modelMag_r IS NOT NULL
LIMIT 1000
"""

# URL encode the query
query_encoded = query.replace("\n", " ").replace(" ", "%20").replace("'", "%27")

# SDSS REST API endpoint
url = f"https://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SQL?format=json&query={query_encoded}"

# Send the HTTP request to SDSS
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Load the JSON data
    data = response.json()
    # Convert the JSON data to a pandas DataFrame
    df = pd.DataFrame(data)
    # Display the first few rows of the dataframe
    print(df.head())
    # Save the data to a CSV file
    df.to_csv("sdss_galaxy_data.csv", index=False)
else:
    print(f"Failed to retrieve data: HTTP {response.status_code}")

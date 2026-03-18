import requests
import os

def get_ml_dataset_ids(limit=1000):
    """Fetches high-resolution PDB IDs suitable for ML."""
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    # Corrected v2 API payload
    query = {
      "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_entry_info.resolution_combined",
              "operator": "less_or_equal",
              "value": 2.0
            }
          },
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "entity_poly.rcsb_sample_sequence_length",
              "operator": "greater_or_equal",
              "value": 50
            }
          },
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "entity_poly.rcsb_sample_sequence_length",
              "operator": "less_or_equal",
              "value": 500
            }
          }
        ]
      },
      "request_options": {
        "paginate": {
          "start": 0,
          "rows": limit
        }
      },
      "return_type": "entry"
    }

    print("Querying RCSB PDB for IDs...")
    response = requests.post(search_url, json=query)
    
    if response.status_code == 200:
        data = response.json()
        if 'result_set' in data:
            # Extract the IDs from the response
            pdb_ids = [item['identifier'] for item in data['result_set']]
            return pdb_ids
        else:
            print("Query succeeded, but no results were found.")
            return []
    else:
        print(f"Error querying API: {response.status_code}")
        print(f"API Response: {response.text}") # Prints the exact syntax error
        return []

def download_cif_files(pdb_ids, download_dir="pdb_dataset"):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    print(f"Starting download of {len(pdb_ids)} files to '{download_dir}'...")
    
    for i, pdb_id in enumerate(pdb_ids):
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        response = requests.get(url)
        
        if response.status_code == 200:
            file_path = os.path.join(download_dir, f"{pdb_id}.cif")
            with open(file_path, 'w') as f:
                f.write(response.text)
            if (i + 1) % 50 == 0:
                print(f"Downloaded {i + 1}/{len(pdb_ids)}...")
        else:
            print(f"Failed to download {pdb_id}")

# Execute the pipeline
dataset_ids = get_ml_dataset_ids(limit=1000)
if dataset_ids:
    print(f"Successfully found {len(dataset_ids)} unique IDs.")
    
    # Uncomment the line below to trigger the actual file downloads
    download_cif_files(dataset_ids)
import requests
import gzip
import os

BENCHMARK_PDBS = [
    "1UBQ", "1CRN", "1MBN", "1EMA",
    "1HHP", "4INS", "1HHO", "1TIM",
]

def download_cif_files(pdb_ids, download_dir="benchmark_dataset"):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    print(f"Starting download of {len(pdb_ids)} benchmark files to '{download_dir}'...")

    for i, pdb_id in enumerate(pdb_ids):
        file_path = os.path.join(download_dir, f"{pdb_id}.cif")

        # Skip if already exists and is valid
        if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
            with open(file_path) as f:
                head = f.read(200)
            if "data_" in head:
                print(f"  Already exists: {pdb_id}.cif")
                continue

        pdb_lower = pdb_id.lower()
        mid = pdb_lower[1:3]

        # Try multiple mirrors in order
        urls = [
            (f"https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb_lower}.cif", False),
            (f"https://data.pdbj.org/pdbjplus/data/pdb/cif/{mid}/{pdb_lower}.cif.gz", True),
            (f"https://ftp.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/{mid}/{pdb_lower}.cif.gz", True),
        ]

        downloaded = False
        for url, is_gzipped in urls:
            try:
                response = requests.get(url, timeout=60, allow_redirects=True)

                if response.status_code != 200:
                    continue

                content = response.content
                if content[:20].strip().startswith(b"<!"):
                    continue

                if is_gzipped:
                    content = gzip.decompress(content)

                text = content.decode("utf-8", errors="ignore")

                if "data_" not in text[:500]:
                    continue

                with open(file_path, 'w') as f:
                    f.write(text)

                print(f"  Downloaded {pdb_id}.cif ({len(text)//1024} KB)")
                downloaded = True
                break

            except Exception:
                continue

        if not downloaded:
            print(f"  Failed to download {pdb_id}")

# Execute the pipeline
print(f"Downloading {len(BENCHMARK_PDBS)} benchmark proteins...")
download_cif_files(BENCHMARK_PDBS)
print("Done.")
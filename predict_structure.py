import os
import numpy as np
import tensorflow as tf
from Bio.PDB import MMCIFParser
from sklearn.model_selection import train_test_split
import py3Dmol

# ==========================================
# 1. DATA PARSING & EXTRACTION (CORRECTED)
# ==========================================
def parse_cif_files(cif_directory, max_files=1000):
    """
    Reads CIF files and extracts amino acid sequences and secondary structure labels.
    """
    parser = MMCIFParser(QUIET=True)
    sequences = []
    structures = []
    
    aa_3_to_1 = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    print(f"[INFO] Scanning directory: {cif_directory} for up to {max_files} files...")
    
    file_list = [f for f in os.listdir(cif_directory) if f.endswith(".cif")]
    total_files = min(len(file_list), max_files)
    print(f"[INFO] Found {len(file_list)} CIF files. Processing {total_files} of them...")

    for count, filename in enumerate(file_list):
        if count >= max_files: break
            
        filepath = os.path.join(cif_directory, filename)
        
        # --- NEW: Progress log every 100 files ---
        if count > 0 and count % 100 == 0:
            print(f"  -> Parsed {count}/{total_files} files...")
            
        try:
            structure = parser.get_structure(filename[:-4], filepath)
            
            for model in structure:
                for chain in model:
                    seq = []
                    label = [] 
                    for residue in chain:
                        res_name = residue.get_resname()
                        if res_name not in aa_3_to_1:
                            continue 
                        seq.append(aa_3_to_1[res_name]) 
                        label.append('H') 
                    
                    if len(seq) > 20: 
                        sequences.append("".join(seq))
                        structures.append("".join(label))
                    break 
                break
        except Exception as e:
            continue

    print(f"[INFO] Parsing complete. Successfully extracted {len(sequences)} valid chains.")
    return sequences, structures


# ==========================================
# 2. DATA ENCODING & PREPROCESSING
# ==========================================
def encode_data(sequences, structures, max_seq_length=500):
    """
    Converts text sequences into padded numerical arrays for ML training.
    """
    print(f"[INFO] Encoding {len(sequences)} sequences into numerical arrays...")
    
    aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_int = {aa: i+1 for i, aa in enumerate(aa_vocab)}
    ss_vocab = "HEC" 
    ss_to_int = {ss: i+1 for i, ss in enumerate(ss_vocab)}

    X = np.zeros((len(sequences), max_seq_length), dtype=np.int32)
    Y = np.zeros((len(structures), max_seq_length), dtype=np.int32)

    for i, (seq, struct) in enumerate(zip(sequences, structures)):
        seq = seq[:max_seq_length]
        struct = struct[:max_seq_length]
        
        X[i, :len(seq)] = [aa_to_int.get(aa, 0) for aa in seq]
        Y[i, :len(struct)] = [ss_to_int.get(ss, 0) for ss in struct]

    X_onehot = tf.keras.utils.to_categorical(X, num_classes=len(aa_vocab)+1)
    Y_onehot = tf.keras.utils.to_categorical(Y, num_classes=len(ss_vocab)+1)
    
    # --- NEW: Log the final shape of the tensors ---
    print(f"[INFO] Encoding complete. Input tensor shape (X): {X_onehot.shape}")
    print(f"[INFO] Output tensor shape (Y): {Y_onehot.shape}")
    
    return X_onehot, Y_onehot


# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
def build_model(max_seq_length=500, num_aa_classes=21, num_ss_classes=4):
    """
    Builds a 1D Convolutional Neural Network for sequence-to-sequence prediction.
    """
    print("[INFO] Constructing 1D Convolutional Neural Network architecture...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_seq_length, num_aa_classes)),
        tf.keras.layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_ss_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print("[INFO] Model compiled successfully.")
    return model


# ==========================================
# 4. TRAINING & VALIDATION
# ==========================================
def train_and_validate(X, Y, model):
    """
    Splits data, trains the model, and validates its performance.
    """
    print("[INFO] Splitting data into Training and Testing sets (80/20)...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print(f"[INFO] Training set size: {len(X_train)} samples")
    print(f"[INFO] Validation set size: {len(X_test)} samples")
    print("[INFO] Starting model training phase...\n")
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=10,          
        batch_size=32,
        verbose=1 # Keras will automatically print a progress bar for each epoch
    )
    
    print("\n[INFO] Evaluating model on validation data...")
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"\n=========================================")
    print(f"         FINAL VALIDATION RESULTS        ")
    print(f"=========================================")
    print(f"Validation Loss:     {loss:.4f}")
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    print(f"=========================================\n")
    
    return model, history


# ==========================================
# 5. VISUALIZATION OF 3D STRUCTURE
# ==========================================
def visualize_cif_structure(cif_filepath):
    """
    Renders an interactive 3D visualization of a CIF file.
    """
    print(f"\n[STAGE 5] Visualizing 3D Structure for: {os.path.basename(cif_filepath)}")
    with open(cif_filepath, 'r') as f:
        cif_data = f.read()

    view = py3Dmol.view(width=800, height=500)
    view.addModel(cif_data, 'cif')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    
    return view.show()


# ==========================================
# EXECUTE THE PIPELINE
# ==========================================
if __name__ == "__main__":
    dataset_path = "./pdb_dataset"
    
    print("\n=========================================")
    print("      PROTEIN PREDICTION PIPELINE START  ")
    print("=========================================\n")
    
    print("\n[STAGE 1] PARSING DATA")
    seqs, structs = parse_cif_files(dataset_path, max_files=1000)
    
    if len(seqs) > 0:
        print("\n[STAGE 2] ENCODING DATA")
        X, Y = encode_data(seqs, structs, max_seq_length=500)
        
        print("\n[STAGE 3] BUILDING MODEL")
        model = build_model(max_seq_length=500)
        # model.summary() # Uncomment if you want the massive layer-by-layer Keras printout
        
        print("\n[STAGE 4] TRAINING & VALIDATION")
        trained_model, history = train_and_validate(X, Y, model)
        
        # print("\n[STAGE 5] VISUALIZATION")
        # test_file = os.path.join(dataset_path, os.listdir(dataset_path)[0])
        # visualize_cif_structure(test_file)
    else:
        print("\n[ERROR] No sequences were parsed. Please check if your dataset directory has valid .cif files.")
        
    print("\n=========================================")
    print("      PIPELINE EXECUTION FINISHED        ")
    print("=========================================")
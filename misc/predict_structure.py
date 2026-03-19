import os
import numpy as np
import tensorflow as tf
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from sklearn.model_selection import train_test_split
import py3Dmol
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA PARSING & EXTRACTION (REAL LABELS)
# ==========================================
def parse_cif_files(cif_directory, max_files=1000):
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
    
    if not os.path.exists(cif_directory):
        print(f"[ERROR] Directory {cif_directory} not found.")
        return [], []

    file_list = [f for f in os.listdir(cif_directory) if f.endswith(".cif")]
    total_files = min(len(file_list), max_files)
    print(f"[INFO] Found {len(file_list)} CIF files. Processing {total_files} of them...")

    for count, filename in enumerate(file_list):
        if count >= max_files: break
        filepath = os.path.join(cif_directory, filename)
        
        if count > 0 and count % 100 == 0:
            print(f"  -> Parsed {count}/{total_files} files...")
            
        try:
            # --- NEW: Extract real secondary structure from CIF metadata ---
            cif_dict = MMCIF2Dict(filepath)
            
            # Map out all Helix positions
            helix_ids = set()
            if '_struct_conf.beg_auth_seq_id' in cif_dict:
                starts = cif_dict['_struct_conf.beg_auth_seq_id']
                ends = cif_dict['_struct_conf.end_auth_seq_id']
                for s, e in zip(starts, ends):
                    try:
                        for idx in range(int(s), int(e) + 1):
                            helix_ids.add(idx)
                    except ValueError: pass # Ignore corrupted or missing IDs
                    
            # Map out all Sheet positions
            sheet_ids = set()
            if '_struct_sheet_range.beg_auth_seq_id' in cif_dict:
                starts = cif_dict['_struct_sheet_range.beg_auth_seq_id']
                ends = cif_dict['_struct_sheet_range.end_auth_seq_id']
                for s, e in zip(starts, ends):
                    try:
                        for idx in range(int(s), int(e) + 1):
                            sheet_ids.add(idx)
                    except ValueError: pass

            # --- Extract sequence and apply real labels ---
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
                        
                        # Get the physical residue number
                        res_seq_num = residue.get_id()[1]
                        
                        # Apply the real label based on the metadata map
                        if res_seq_num in helix_ids:
                            label.append('H')
                        elif res_seq_num in sheet_ids:
                            label.append('E')
                        else:
                            label.append('C') # Random Coil / Loop
                    
                    if len(seq) > 20: 
                        # Only append if it's a real protein with some actual structure
                        if 'H' in label or 'E' in label:
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
    
    print(f"[INFO] Encoding complete. Input tensor shape (X): {X_onehot.shape}")
    print(f"[INFO] Output tensor shape (Y): {Y_onehot.shape}")
    
    return X_onehot, Y_onehot, aa_vocab, ss_vocab


# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
def build_model(max_seq_length=500, num_aa_classes=21, num_ss_classes=4):
    print("[INFO] Constructing 1D Convolutional Neural Network architecture...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_seq_length, num_aa_classes)),
        tf.keras.layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_ss_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("[INFO] Model compiled successfully.")
    return model


# ==========================================
# 4. TRAINING & VALIDATION
# ==========================================
def train_and_validate(X, Y, model):
    print("[INFO] Splitting data into Training and Testing sets (80/20)...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print(f"[INFO] Training set size: {len(X_train)} samples")
    print(f"[INFO] Validation set size: {len(X_test)} samples")
    print("[INFO] Starting model training phase...\n")
    
    history = model.fit(
        X_train, Y_train, validation_data=(X_test, Y_test),
        epochs=15, batch_size=32, verbose=1 
    )
    
    print("\n[INFO] Evaluating model on validation data...")
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"\n=========================================")
    print(f"         FINAL VALIDATION RESULTS        ")
    print(f"=========================================")
    print(f"Validation Loss:     {loss:.4f}")
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    print(f"=========================================\n")
    
    return model, X_test, Y_test


# ==========================================
# 5. SAVE OUTPUTS (SUMMARY & IMAGE)
# ==========================================
def generate_and_save_outputs(model, X_sample, y_sample, ss_vocab, output_dir="./outputs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n[INFO] Generating predictions for sample output...")
    
    sample_input = np.expand_dims(X_sample, axis=0)
    prediction = model.predict(sample_input, verbose=0) 
    
    pred_indices = np.argmax(prediction[0], axis=-1)
    true_indices = np.argmax(y_sample, axis=-1)
    
    ss_map = {i+1: char for i, char in enumerate(ss_vocab)}
    ss_map[0] = '-' 
    
    pred_str = "".join([ss_map.get(idx, '-') for idx in pred_indices if idx != 0])
    true_str = "".join([ss_map.get(idx, '-') for idx in true_indices if idx != 0])
    
    summary_path = os.path.join(output_dir, "prediction_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== PROTEIN SECONDARY STRUCTURE PREDICTION SUMMARY ===\n\n")
        f.write(f"Sequence Length: {len(true_str)} amino acids\n\n")
        f.write("TRUE STRUCTURE (Ground Truth):\n")
        f.write(true_str + "\n\n")
        f.write("PREDICTED STRUCTURE (Model Output):\n")
        f.write(pred_str + "\n\n")
        f.write("Legend: H = Helix, E = Sheet, C = Coil, - = Padding\n")
    print(f"[INFO] Prediction summary saved to: {summary_path}")

    print("[INFO] Generating visual map of the predicted structure...")
    image_path = os.path.join(output_dir, "predicted_structure_map.png")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    colors = {'H': 'red', 'E': 'blue', 'C': 'gray', '-': 'white'}
    
    x_vals = range(len(true_str))
    
    # Plot True
    true_c = [colors[char] for char in true_str]
    ax1.bar(x_vals, [1]*len(true_str), color=true_c, width=1.0)
    ax1.set_yticks([])
    ax1.set_title("True Secondary Structure")

    # Plot Predicted
    pred_c = [colors[char] for char in pred_str]
    ax2.bar(x_vals, [1]*len(pred_str), color=pred_c, width=1.0)
    ax2.set_yticks([])
    ax2.set_title("Predicted Secondary Structure")
    
    ax2.set_xlim(0, len(true_str))
    ax2.set_xlabel("Amino Acid Position")
    
    plt.tight_layout()
    plt.savefig(image_path, dpi=300)
    plt.close()
    
    print(f"[INFO] Structural prediction image saved to: {image_path}")


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
        X, Y, aa_vocab, ss_vocab = encode_data(seqs, structs, max_seq_length=500)
        
        print("\n[STAGE 3] BUILDING MODEL")
        model = build_model(max_seq_length=500)
        
        print("\n[STAGE 4] TRAINING & VALIDATION")
        trained_model, X_test, Y_test = train_and_validate(X, Y, model)
        
        print("\n[STAGE 5] SAVING OUTPUTS")
        generate_and_save_outputs(trained_model, X_test[0], Y_test[0], ss_vocab)
        
    else:
        print("\n[ERROR] No sequences were parsed.")
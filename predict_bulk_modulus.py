#!/usr/bin/env python3
"""
Predict bulk modulus for a single CIF file using a trained classifier at timestep=0.

Usage:
    python predict_bulk_modulus.py path/to/structure.cif \
        --ckpt checkpoints/bulk_modulus_classifier_mixed_16_large/best.pt
"""

import argparse
import sys
import torch
from pymatgen.io.cif import CifParser

# Import prediction utilities from the existing testing script
try:
    from test_bulk_modulus import load_classifier, predict_with_classifier
except ImportError:
    print("Error: Could not import from test_bulk_modulus.py. Make sure it is in the same directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Predict bulk modulus from a CIF file at timestep=0.")
    parser.add_argument("cif_file", type=str, help="Path to the input CIF file")
    parser.add_argument(
        "--ckpt", 
        type=str, 
        default="checkpoints/bulk_modulus_classifier_mixed_16_large/best.pt", 
        help="Path to the trained classifier checkpoint"
    )
    parser.add_argument(
        "--predictor_type", 
        type=str, 
        choices=["mlp", "lora", "lora_mlp"], 
        default="lora_mlp", 
        help="Type of the predictor checkpoint (default: lora_mlp)"
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Parsing CIF file: {args.cif_file}")
    try:
        # We use primitive=True as in test_bulk_modulus.py
        struct = CifParser(args.cif_file).parse_structures(primitive=True)[0]
        print(f"Structure loaded successfully: {struct.composition.reduced_formula} ({len(struct)} atoms)")
    except Exception as e:
        print(f"Failed to parse CIF file: {e}")
        sys.exit(1)
        
    print(f"\nLoading classifier ({args.predictor_type}) from {args.ckpt}...")
    try:
        classifier = load_classifier(args.ckpt, args.predictor_type, device)
        classifier.eval()
        print(f"Classifier loaded successfully on {device}")
    except Exception as e:
        print(f"Failed to load classifier model: {e}")
        sys.exit(1)
        
    print("\nPredicting bulk modulus at timestep=0...")
    try:
        # predict_with_classifier handles converting primitive structure to ChemGraph and passing timestep=0
        pred_val = predict_with_classifier(classifier, struct, device)
        print("="*50)
        print(f"Predicted Bulk Modulus: {pred_val:.2f} GPa")
        print("="*50)
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

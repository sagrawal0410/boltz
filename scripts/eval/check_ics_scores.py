import json
from pathlib import Path

def check_ics_scores(eval_results_dir):
    """
    Check which structure JSON files have non-zero ics scores.
    
    Args:
        eval_results_dir: Path to directory containing JSON evaluation results
    """
    eval_dir = Path(eval_results_dir)
    json_files = list(eval_dir.glob("*.json"))
    
    # Filter out ligand JSON files
    structure_json_files = [f for f in json_files if "_ligand.json" not in f.name]
    
    non_zero_ics = []
    zero_ics = []
    missing_ics = []
    
    print(f"Checking {len(structure_json_files)} structure JSON files...\n")
    
    for json_file in structure_json_files:
        try:
            with json_file.open("r") as f:
                eval_data = json.load(f)
            
            if "ics" not in eval_data:
                missing_ics.append(json_file.name)
            elif eval_data["ics"] != 0.0:
                non_zero_ics.append((json_file.name, eval_data["ics"]))
            else:
                zero_ics.append(json_file.name)
                
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    # Print results
    print(f"Files with non-zero ics scores: {len(non_zero_ics)}")
    if non_zero_ics:
        print("\nNon-zero ics scores:")
        for filename, ics_value in non_zero_ics:
            print(f"  {filename}: {ics_value}")
    
    print(f"\nFiles with zero ics scores: {len(zero_ics)}")
    print(f"Files missing ics field: {len(missing_ics)}")
    
    if missing_ics:
        print("\nFiles missing ics field:")
        for filename in missing_ics[:10]:  # Show first 10
            print(f"  {filename}")
        if len(missing_ics) > 10:
            print(f"  ... and {len(missing_ics) - 10} more")
    
    return {
        "non_zero": non_zero_ics,
        "zero": zero_ics,
        "missing": missing_ics
    }

if __name__ == "__main__":
    import sys
    
    eval_dir = sys.argv[1] if len(sys.argv) > 1 else "./eval_results"
    results = check_ics_scores(eval_dir)
import os
import json
import subprocess

def run_script(script_name):
    print(f"==================================================")
    print(f"Running {script_name}...")
    print(f"==================================================")
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    subprocess.run([sys.executable, script_path], check=True)

def generate_results_md():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    md_path = os.path.join(results_dir, "results.md")
    
    with open(md_path, "w") as f:
        f.write("# PeerReview AI Evaluation Results\n\n")
        
        # 1. Guardrail Leak Rate
        f.write("## 1. Guardrail Leak Rate (Reviewer Node Red-Team)\n\n")
        try:
            with open(os.path.join(results_dir, "guardrail_leak_rate.json")) as jf:
                leak_data = json.load(jf)["summary"]
            f.write("| Attack style | Without guardrail | With guardrail |\n")
            f.write("|--------------|-------------------|----------------|\n")
            for style, stats in leak_data.items():
                if style != "TOTAL":
                    f.write(f"| {style} | {stats['without_guardrail']} | {stats['with_guardrail']} |\n")
            f.write(f"| **TOTAL** | **{leak_data['TOTAL']['without_guardrail']}** | **{leak_data['TOTAL']['with_guardrail']}** |\n\n")
        except Exception as e:
            f.write(f"*Error loading leak rate data: {e}*\n\n")
            
        # 2. Retrieval Precision
        f.write("## 2. Retrieval Precision (RAG Pipeline)\n\n")
        try:
            with open(os.path.join(results_dir, "retrieval_precision.json")) as jf:
                rag_data = json.load(jf)["summary"]
            f.write("| Method | Precision@3 | MRR |\n")
            f.write("|--------|-------------|-----|\n")
            f.write(f"| Bi-encoder only | {rag_data['bi_encoder']['precision_at_3']:.2f} | {rag_data['bi_encoder']['mrr']:.2f} |\n")
            f.write(f"| Full RAG pipeline | {rag_data['full_pipeline']['precision_at_3']:.2f} | {rag_data['full_pipeline']['mrr']:.2f} |\n\n")
        except Exception as e:
            f.write(f"*Error loading retrieval data: {e}*\n\n")
            
        # 3. Recommender vs Baseline
        f.write("## 3. Recommender vs Baseline (Synthetic Sessions)\n\n")
        try:
            with open(os.path.join(results_dir, "recommender_vs_baseline.json")) as jf:
                rec_data = json.load(jf)["summary"]
            f.write("| Method | Top-1 accuracy | Top-2 accuracy |\n")
            f.write("|--------|----------------|----------------|\n")
            for method in ["Random", "Longest-untouched", "GMM+XGBoost"]:
                stats = rec_data[method]["overall"]
                f.write(f"| {method} | {stats['top1']:.2f} | {stats['top2']:.2f} |\n")
                
            f.write("\n### By Skill Level\n\n")
            for method in ["Random", "Longest-untouched", "GMM+XGBoost"]:
                f.write(f"**{method}**\n")
                for lvl in ["novice", "intermediate", "advanced"]:
                    lstats = rec_data[method]["by_level"][lvl]
                    f.write(f"- {lvl.capitalize()}: Top-1={lstats['top1']:.2f}, Top-2={lstats['top2']:.2f}\n")
                f.write("\n")
                
        except Exception as e:
            f.write(f"*Error loading recommender data: {e}*\n\n")
            
    print(f"\nSuccessfully generated {md_path}")

if __name__ == "__main__":
    import sys
    run_script("test_guardrail_leak_rate.py")
    run_script("test_retrieval_precision.py")
    run_script("test_recommender_vs_baseline.py")
    generate_results_md()

import os
import sys
import json
import random
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from ml_engine import rank_sections
from models import SectionType

class MockSection:
    def __init__(self, id, type_val, words, rev_count, days_since):
        self.id = id
        self.section_type = type_val
        self.content = "word " * words
        self.revision_count = rev_count
        self.last_revised_at = datetime.utcnow() - timedelta(days=days_since)

def generate_synthetic_session(skill_level: str, seed: int):
    random.seed(seed)
    
    # 5 sections per draft
    sections_types = [
        SectionType.INTRO, 
        SectionType.RELATED_WORK, 
        SectionType.METHODS, 
        SectionType.RESULTS, 
        SectionType.DISCUSSION
    ]
    
    # Baseline stats by skill
    if skill_level == "novice":
        days_base = [30, 45, 10, 5, 60]
        rev_base = [2, 1, 5, 8, 1]
    elif skill_level == "intermediate":
        days_base = [10, 15, 12, 8, 20]
        rev_base = [5, 4, 6, 7, 3]
    else: # advanced
        days_base = [2, 5, 3, 2, 4]
        rev_base = [12, 8, 15, 14, 10]
        
    sections = []
    
    # We want one section to clearly be the "most-needed". 
    # Let's say we artificially degrade one section's quality.
    target_idx = random.randint(0, 4)
    target_type = sections_types[target_idx]
    
    for i, stype in enumerate(sections_types):
        words = random.randint(200, 800)
        
        rev_count = rev_base[i] + random.randint(-1, 2)
        days = days_base[i] + random.randint(-2, 5)
        
        # If it's the target, make it look very abandoned or neglected
        if i == target_idx:
            days += 20
            rev_count = max(0, rev_count - 3)
            words = random.randint(50, 150) # Very short
            
        sec = MockSection(id=i+1, type_val=stype.value, words=words, rev_count=max(0, rev_count), days_since=max(1, days))
        sections.append(sec)
        
    return sections, target_type.value

def run_evaluation():
    levels = ["novice", "intermediate", "advanced"]
    
    results = []
    
    stats = {
        "Random": {"top1": 0, "top2": 0, "total": 0, "by_level": {l: {"top1":0, "top2":0, "total":0} for l in levels}},
        "Longest-untouched": {"top1": 0, "top2": 0, "total": 0, "by_level": {l: {"top1":0, "top2":0, "total":0} for l in levels}},
        "GMM+XGBoost": {"top1": 0, "top2": 0, "total": 0, "by_level": {l: {"top1":0, "top2":0, "total":0} for l in levels}}
    }
    
    print("Running Recommender vs Baseline Evaluation...")
    
    for level in levels:
        for seed in range(50):
            sections, ground_truth = generate_synthetic_session(level, seed + hash(level))
            
            # Baseline 1: Random
            random.seed(seed + hash(level) + 100)
            rand_ranked = list(sections)
            random.shuffle(rand_ranked)
            rand_top1 = rand_ranked[0].section_type == ground_truth
            rand_top2 = rand_ranked[0].section_type == ground_truth or rand_ranked[1].section_type == ground_truth
            
            # Baseline 2: Longest-untouched
            longest_ranked = sorted(sections, key=lambda s: s.last_revised_at)
            long_top1 = longest_ranked[0].section_type == ground_truth
            long_top2 = longest_ranked[0].section_type == ground_truth or longest_ranked[1].section_type == ground_truth
            
            # Model: GMM + XGBoost
            model_ranked = rank_sections(draft=None, sections=sections, archetype_id=1)
            # If nothing returned (e.g. all pruned), fallback to random for the model
            if len(model_ranked) == 0:
                model_top1 = False
                model_top2 = False
                mod_res = []
            else:
                mod_res = [s["section_type"] for s in model_ranked]
                model_top1 = model_ranked[0]["section_type"] == ground_truth
                model_top2 = model_ranked[0]["section_type"] == ground_truth or (len(model_ranked)>1 and model_ranked[1]["section_type"] == ground_truth)
                
            results.append({
                "level": level,
                "seed": seed,
                "ground_truth": ground_truth,
                "random_top1": rand_top1,
                "longest_top1": long_top1,
                "model_top1": model_top1,
                "model_res": mod_res
            })
            
            for m, (t1, t2) in zip(["Random", "Longest-untouched", "GMM+XGBoost"], 
                                   [(rand_top1, rand_top2), (long_top1, long_top2), (model_top1, model_top2)]):
                stats[m]["total"] += 1
                stats[m]["by_level"][level]["total"] += 1
                if t1:
                    stats[m]["top1"] += 1
                    stats[m]["by_level"][level]["top1"] += 1
                if t2:
                    stats[m]["top2"] += 1
                    stats[m]["by_level"][level]["top2"] += 1
                    
    # Print summary
    print("\nMethod                  | Top-1 accuracy | Top-2 accuracy")
    print("-" * 65)
    
    summary_out = {}
    for method in ["Random", "Longest-untouched", "GMM+XGBoost"]:
        t1_pct = stats[method]["top1"] / stats[method]["total"]
        t2_pct = stats[method]["top2"] / stats[method]["total"]
        print(f"{method.ljust(23)} |    {t1_pct:.2f}        |    {t2_pct:.2f}")
        
        lvl_stats = {}
        for lvl in levels:
            l_t1 = stats[method]["by_level"][lvl]["top1"] / stats[method]["by_level"][lvl]["total"]
            l_t2 = stats[method]["by_level"][lvl]["top2"] / stats[method]["by_level"][lvl]["total"]
            lvl_stats[lvl] = {"top1": l_t1, "top2": l_t2}
            
        summary_out[method] = {
            "overall": {"top1": t1_pct, "top2": t2_pct},
            "by_level": lvl_stats
        }

    output = {
        "summary": summary_out,
        "details": results
    }
    
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results", "recommender_vs_baseline.json"), "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    run_evaluation()

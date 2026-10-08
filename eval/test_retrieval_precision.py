import os
import sys
import json
import asyncio

# Add backend to path so we can import from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from rag_engine import search, get_chroma_collection
from config import settings

def precision_at_k(returned_doc_ids, relevant_doc_ids, k=3):
    top_k = returned_doc_ids[:k]
    relevant_set = set(relevant_doc_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / k

def mrr_at_k(returned_doc_ids, relevant_doc_ids, k=3):
    top_k = returned_doc_ids[:k]
    relevant_set = set(relevant_doc_ids)
    for i, doc_id in enumerate(top_k):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0

def get_cache():
    cache_path = os.path.join(os.path.dirname(__file__), "eval_rag_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    cache_path = os.path.join(os.path.dirname(__file__), "eval_rag_cache.json")
    with open(cache_path, "w") as f:
        json.dump(cache, f)

async def run_evaluation():
    queries_file = os.path.join(os.path.dirname(__file__), "labeled_queries.json")
    with open(queries_file, "r") as f:
        queries = json.load(f)
        
    collection = get_chroma_collection()
    
    results_out = []
    
    total_prec_bi = 0.0
    total_mrr_bi = 0.0
    total_prec_full = 0.0
    total_mrr_full = 0.0
    
    print("Running Retrieval Precision Evaluation...")
    cache = get_cache()
    
    for idx, item in enumerate(queries):
        print(f"Processing query {idx+1}/{len(queries)}")
        
        query = item["query"]
        relevant = item["relevant_doc_ids"]
        
        cache_key = item["id"]
        if cache_key in cache:
            res = cache[cache_key]
            results_out.append(res)
            total_prec_bi += res["bi_prec_3"]
            total_mrr_bi += res["bi_mrr"]
            total_prec_full += res["full_prec_3"]
            total_mrr_full += res["full_mrr"]
            continue
            
        async def call_with_retry(func, *args, **kwargs):
            from google.genai import errors
            while True:
                try:
                    return await func(*args, **kwargs)
                except errors.ClientError as e:
                    if "429" in str(e) or e.code == 429:
                        print("Rate limit hit! Sleeping for 60 seconds...")
                        await asyncio.sleep(60)
                    else:
                        raise e

        await asyncio.sleep(4)
        
        # 1. Bi-encoder only (raw ChromaDB query)
        chroma_res = collection.query(
            query_texts=[query],
            n_results=3
        )
        if chroma_res and chroma_res.get("ids") and len(chroma_res["ids"]) > 0:
            bi_returned = [os.path.basename(path) for path in chroma_res["ids"][0]]
        else:
            bi_returned = []
            
        prec_bi = precision_at_k(bi_returned, relevant, 3)
        mrr_bi = mrr_at_k(bi_returned, relevant, 3)
        
        # 2. Full RAG pipeline (search function)
        rag_res = await call_with_retry(search, query)
        # Note: the search results might be returned as pydantic models or dicts.
        try:
            full_returned = [os.path.basename(r.source) for r in rag_res]
        except AttributeError:
            full_returned = [os.path.basename(r["source"]) for r in rag_res]
        
        prec_full = precision_at_k(full_returned, relevant, 3)
        mrr_full = mrr_at_k(full_returned, relevant, 3)
        
        total_prec_bi += prec_bi
        total_mrr_bi += mrr_bi
        total_prec_full += prec_full
        total_mrr_full += mrr_full
        
        res_entry = {
            "id": item["id"],
            "query": query,
            "relevant_doc_ids": relevant,
            "bi_encoder_returned": bi_returned,
            "bi_prec_3": prec_bi,
            "bi_mrr": mrr_bi,
            "full_pipeline_returned": full_returned,
            "full_prec_3": prec_full,
            "full_mrr": mrr_full
        }
        
        results_out.append(res_entry)
        cache[cache_key] = res_entry
        save_cache(cache)
        
    avg_prec_bi = total_prec_bi / len(queries)
    avg_mrr_bi = total_mrr_bi / len(queries)
    avg_prec_full = total_prec_full / len(queries)
    avg_mrr_full = total_mrr_full / len(queries)

    print("\nMethod              | Precision@3 | MRR")
    print("-" * 50)
    print(f"Bi-encoder only      |   {avg_prec_bi:.2f}      | {avg_mrr_bi:.2f}")
    print(f"Full RAG pipeline     |   {avg_prec_full:.2f}      | {avg_mrr_full:.2f}")
    
    output = {
        "summary": {
            "bi_encoder": {
                "precision_at_3": avg_prec_bi,
                "mrr": avg_mrr_bi
            },
            "full_pipeline": {
                "precision_at_3": avg_prec_full,
                "mrr": avg_mrr_full
            }
        },
        "details": results_out
    }
    
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results", "retrieval_precision.json"), "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_evaluation())

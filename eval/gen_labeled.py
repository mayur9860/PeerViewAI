import json
import os

queries = [
    {"query": "how do I frame a contribution without overclaiming results", "relevant_doc_ids": ["01_framing_contributions_intro_cs.md", "02_avoiding_overclaiming_general.md", "09_contribution_patterns_general.md"], "field": "general", "section_type": "intro"},
    {"query": "synthesize related work instead of just listing papers", "relevant_doc_ids": ["03_related_work_synthesis_cs.md"], "field": "cs", "section_type": "related_work"},
    {"query": "writing clear methods sections for biology", "relevant_doc_ids": ["04_methods_clarity_biology.md"], "field": "biology", "section_type": "methods"},
    {"query": "how to separate discussion from results", "relevant_doc_ids": ["05_discussion_vs_results_general.md"], "field": "general", "section_type": "results"},
    {"query": "using hedging language appropriately", "relevant_doc_ids": ["06_hedging_language_general.md"], "field": "general", "section_type": "discussion"},
    {"query": "anticipating reviewer objections in computer science", "relevant_doc_ids": ["07_reviewer_objections_cs.md"], "field": "cs", "section_type": "discussion"},
    {"query": "structure of an economics introduction", "relevant_doc_ids": ["08_intro_structure_economics.md"], "field": "economics", "section_type": "intro"},
    {"query": "writing explicit contribution statements", "relevant_doc_ids": ["09_contribution_patterns_general.md"], "field": "general", "section_type": "intro"},
    {"query": "effective citation practices", "relevant_doc_ids": ["10_citation_practices_general.md"], "field": "general", "section_type": "related_work"},
    {"query": "building narrative in biology results", "relevant_doc_ids": ["11_results_narrative_biology.md"], "field": "biology", "section_type": "results"},
    {"query": "reproducibility in computer science methods", "relevant_doc_ids": ["12_methods_reproducibility_cs.md"], "field": "cs", "section_type": "methods"},
    {"query": "avoiding structural drift", "relevant_doc_ids": ["13_structure_drifting_general.md"], "field": "general", "section_type": "methods"},
    {"query": "managing jargon in academic writing", "relevant_doc_ids": ["14_jargon_management_general.md"], "field": "general", "section_type": "intro"},
    {"query": "getting discussion length right", "relevant_doc_ids": ["15_discussion_length_general.md"], "field": "general", "section_type": "discussion"},
    {"query": "controlling sentence length", "relevant_doc_ids": ["16_sentence_length_general.md"], "field": "general", "section_type": "intro"},
    {"query": "self-citation ethics and strategy", "relevant_doc_ids": ["17_self_citation_ethics.md"], "field": "general", "section_type": "related_work"},
    {"query": "making figures work in results", "relevant_doc_ids": ["18_results_figures_general.md"], "field": "general", "section_type": "results"},
    {"query": "writing effective conclusions", "relevant_doc_ids": ["19_conclusion_writing_general.md"], "field": "general", "section_type": "conclusion"},
    {"query": "strategic revision for academic papers", "relevant_doc_ids": ["20_revision_strategy_general.md"], "field": "general", "section_type": "methods"},
    {"query": "overclaiming in biology results", "relevant_doc_ids": ["21_overclaiming_results_biology.md"], "field": "biology", "section_type": "results"},
    {"query": "identifying and articulating the research gap", "relevant_doc_ids": ["22_related_work_gap_cs.md"], "field": "cs", "section_type": "related_work"},
    {"query": "statistical methods in economics papers", "relevant_doc_ids": ["23_methods_statistical_economics.md"], "field": "economics", "section_type": "methods"},
    {"query": "opening your paper with a strong hook", "relevant_doc_ids": ["24_intro_hook_general.md"], "field": "general", "section_type": "intro"},
    {"query": "discussing mechanisms in biology papers", "relevant_doc_ids": ["25_discussion_mechanisms_biology.md"], "field": "biology", "section_type": "discussion"},
    {"query": "academic paragraph structure", "relevant_doc_ids": ["26_paragraph_structure_general.md"], "field": "general", "section_type": "intro"},
    {"query": "writing abstracts that get papers read", "relevant_doc_ids": ["27_abstract_writing_general.md"], "field": "general", "section_type": "intro"},
    {"query": "designing effective results tables", "relevant_doc_ids": ["28_table_design_general.md"], "field": "general", "section_type": "results"},
    {"query": "writing substantive future work sections", "relevant_doc_ids": ["29_future_work_general.md"], "field": "general", "section_type": "conclusion"},
    {"query": "preparing your paper for peer review", "relevant_doc_ids": ["30_peer_review_response_general.md"], "field": "general", "section_type": "discussion"},
    {"query": "achieving clarity in complex methods", "relevant_doc_ids": ["31_writing_methods_clarity.md"], "field": "general", "section_type": "methods"},
    {"query": "presenting negative results honestly", "relevant_doc_ids": ["32_handling_negative_results.md"], "field": "general", "section_type": "results"},
    {"query": "writing for interdisciplinary audiences", "relevant_doc_ids": ["33_writing_for_interdisciplinary.md"], "field": "general", "section_type": "intro"},
    {"query": "identification strategy presentation", "relevant_doc_ids": ["34_economics_identification.md"], "field": "economics", "section_type": "methods"},
    {"query": "experimental design in cs papers", "relevant_doc_ids": ["35_cs_experimental_design.md"], "field": "cs", "section_type": "methods"},
    {"query": "preparing for the rebuttal process", "relevant_doc_ids": ["36_writing_rebuttals.md"], "field": "general", "section_type": "discussion"},
    {"query": "interpreting computational results", "relevant_doc_ids": ["37_results_interpretation_cs.md"], "field": "cs", "section_type": "results"},
    {"query": "calibrating literature review depth", "relevant_doc_ids": ["38_lit_review_depth.md"], "field": "general", "section_type": "related_work"},
    {"query": "data visualization best practices", "relevant_doc_ids": ["39_data_visualization_general.md"], "field": "general", "section_type": "results"},
    {"query": "presenting theoretical frameworks", "relevant_doc_ids": ["40_theoretical_framework.md"], "field": "general", "section_type": "methods"},
    {"query": "general tips for hedging and reviewer response", "relevant_doc_ids": ["06_hedging_language_general.md", "30_peer_review_response_general.md"], "field": "general", "section_type": "discussion"},
]

for i, q in enumerate(queries, 1):
    q["id"] = f"q_{i:03d}"

output_dir = r"c:\projects\ReviewAI\peerreview_ai\eval"
with open(os.path.join(output_dir, "labeled_queries.json"), "w") as f:
    json.dump(queries, f, indent=2)

print(f"Generated {len(queries)} labeled queries.")

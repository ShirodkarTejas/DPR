@echo off
python dense_retriever.py ^
model_file="G:\learning\iisc dl\capstone\DPR\downloads\checkpoint\retriever\single-adv-hn\nq\bert-base-encoder.cp" ^
qa_dataset=nq_test ^
ctx_datatsets=[dpr_wiki] ^
encoded_ctx_files=["G:\learning\iisc dl\capstone\DPR\downloads\data\retriever_results\nq\single-adv-hn\wikipedia_passages_*"] ^
out_file="G:\learning\iisc dl\capstone\DPR\nq_test_results.json" 
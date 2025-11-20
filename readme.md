# Measuring the Quality of AI-Generated Clinical Notes: A Systematic Review and Experimental Benchmark of Evaluation Methods

This repository contains the code used in the study:

Dahlberg A., Käenniemi T., Winther-Jensen T., Tapiola O., Luisto R., Puranen T., Gordon M., Sanmark E. & Vartiainen V. (2025). *Measuring the Quality of AI-Generated Clinical Notes: A Systematic Review and Experimental Benchmark of Evaluation Methods*. Preprint, DOI to be added.

## Overview

This project reviews and benchmarks methods for evaluating AI-generated medical texts. The code includes:

* Overlap based metrics for text quality
* Embedding based similarity measures
* LLM as a judge pipelines with prompt templates
* A results notebook for statistics and figures

## Repository Structure

```
SHARE_ARTICLE_1/
├─ cases/
│  ├─ english_cases/
│  ├─ finnish_cases/
│  ├─ swedish_cases/
├─ evaluators/
│  ├─ embedding/
│  ├─ LLMs/
│  └─ overlap/
├─ requirements.txt
├─ run_open.py
├─ run.py
├─ results/
│  ├─ figures/
│  ├─ raw_results/
│  │  ├─ LLMs/
│  │  └─ overlap_semantic/
│  └─ tables/
├─ readme.md
└─ results_visualizations.ipynb
```

## Models, Versions, and Acquisition Dates

**GPT-3.5** - deployment: gpt-3.5, provider: Azure OpenAI (Sweden Central), API version: 2024-12-01-preview, acquisition date: 2025-05-27  
*Model version:* 2024-01-25

**GPT-4** - deployment: gpt-4, provider: Azure OpenAI (Sweden Central), API version: 2024-12-01-preview, acquisition date: 2025-05-27  
*Model version:* 2024-04-09

**GPT-4o** - deployment: gpt-4o, provider: Azure OpenAI (Sweden Central), API version: 2024-12-01-preview, acquisition date: 2025-06-10  
*Model version:* 2024-11-20

**GPT-4.1** - deployment: gpt-4.1, provider: Azure OpenAI (Sweden Central), API version: 2024-12-01-preview, acquisition date: 2025-06-10  
*Model version:* 2025-04-14

**GPT-5 (o3)** - deployment: o3, provider: Azure OpenAI (Sweden Central), API version: 2024-12-01-preview, acquisition date: 2025-08-18  
*Model version:* 2025-04-16

**GPT-5-mini (o3-mini)** - deployment: o3-mini, provider: Azure OpenAI (Sweden Central), API version: 2024-12-01-preview, acquisition date: 2025-08-18  
*Model version:* 2025-01-31

**GPT-5-nano (o4-mini)** - deployment: o4-mini, provider: Azure OpenAI (Sweden Central), API version: 2024-12-01-preview, acquisition date: 2025-08-18  
*Model version:* 2025-04-16

**DeepSeek-R1-0528** - provider: Azure AI Inference SDK (Sweden Central), API version: 2024-05-01-preview, acquisition date: 2025-06-10  
*Model version:* 1

**PORO-34B-Q4_K_M** - variant: 4-bit medium quantization, filename: poro-34b.Q4_K_M.gguf, provider: llama.cpp via HuggingFace (TheBloke/Poro-34B-GGUF), context: 2048 tokens, acquisition date: 2025-06-18  
*Model version:* N/A (local quantized model)

**PORO-34B-Q2_K** - variant: 2-bit quantization, filename: poro-34b.Q2_K.gguf, provider: llama.cpp via HuggingFace (TheBloke/Poro-34B-GGUF), context: 2048 tokens, acquisition date: 2025-06-18  
*Model version:* N/A (local quantized model)

**Gemma 3n:e2b** - variant: effective 2B parameters (Q4_K_M quantization), provider: Meta via Ollama, acquisition date: 2025-07-10  
*Model version:* N/A (local Ollama model)

**Gemma 3n:e4b** - variant: effective 4B parameters (Q4_K_M quantization), provider: Meta via Ollama, acquisition date: 2025-07-10  
*Model version:* N/A (local Ollama model)

**gpt-oss-20b** - variant: base 20B parameters (FP16), provider: OpenAI via Hugging Face, acquisition date: 2025-09-12
*Model version:* N/A (local download)

**gpt-oss-120b** - variant: base 120B parameters (FP16), provider: OpenAI via Hugging Face, acquisition date: 2025-09-12
*Model version:*  N/A (local download)

**BERTScore (RoBERTa)** - variant: roberta-large_L17_noidf with baseline rescaling for EN, provider: Hugging Face (evaluate + transformers), acquisition date: 2025-08-18
*Model version:*  N/A (local install)

**BERTScore (XLM-RoBERTa base)** - variant: xlm-roberta-base, provider: Hugging Face, acquisition date: 2025-08-18
*Model version:* N/A (local install)

**BLEURT, variant: bleurt-base-128** - provider: Elron via Hugging Face, acquisition date: 2025-08-18
*Model version:* N/A 

**Text Embedding 3 Large** - version: text-embedding-3-large, provider: OpenAI, acquisition date: 2025-07-07
*Model version:* N/A

## Notes on Clinical Safety

This repository is for research purposes only, and the medical cases are fictional cases part of the MedBench-research at Karolinska Institutet (Gordon M. et al.).

## Acknowledgements

The authors gratefully acknowledge the creators of the original discharge summaries included in the MedBench dataset. This work was supported by Business Finland (1562/31/2024) through the GenAID research project. AD acknowledges Frans Wilhelm och Waldermar von Frenckells understödsfond and The Finnish Medical Foundation for the award of a personal grant. ES acknowledges Finska Läkaresällskapet, The Finnish Medical Foundation and, Liv och Hälsä Medical Support Association.

## References

Here it is as a Markdown bullet list, alphabetized and unnumbered:

*Google LLC. Python ROUGE implementation. PyPI* (2025). Available from: [https://pypi.org/project/rouge-score/](https://pypi.org/project/rouge-score/)

*Google LLC. rouge-score. PyPI* (2025). Available from: [https://pypi.org/project/rouge-score/](https://pypi.org/project/rouge-score/)

*Harris CR, Millman KJ, van der Walt SJ, et al. Array programming with NumPy. Nature 585*, 357–362 (2020). [https://doi.org/10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)

*Hunter JD. Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering 9*, 90–95 (2007). [https://doi.org/10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)

Jinja [software on the Internet]. Available from: [https://pypi.org/project/Jinja2/] (https://pypi.org/project/Jinja2/)

*LangChain Developers. LangChain. PyPI* (2025). Available from: [https://pypi.org/project/langchain/](https://pypi.org/project/langchain/)

*Lin C-Y. ROUGE: A Package for Automatic Evaluation of Summaries. Text Summarization Branches Out (ACL)*, 74–81 (2004). [https://aclanthology.org/W04-1013](https://aclanthology.org/W04-1013)

*Microsoft Corporation. Microsoft Azure SDK for Python. PyPI* (2025). Available from: [https://pypi.org/project/azure/](https://pypi.org/project/azure/)

*Pallets team. Jinja. PyPI* (2025). Available from: [https://pypi.org/project/Jinja2/](https://pypi.org/project/Jinja2/)

*Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12*, 2825–2830 (2011). [http://www.jmlr.org/papers/v12/pedregosa11a.html](http://www.jmlr.org/papers/v12/pedregosa11a.html)

*Post M. A Call for Clarity in Reporting BLEU Scores. Proceedings of the Third Conference on Machine Translation: Research Papers (WMT)* (2018). [https://doi.org/10.18653/v1/W18-6319](https://doi.org/10.18653/v1/W18-6319)

*Virtanen P, Gommers R, Oliphant TE, et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods 17*, 261–272 (2020). [https://doi.org/10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)

*Waskom ML. seaborn: statistical data visualization. Journal of Open Source Software 6*, 3021 (2021). [https://doi.org/10.21105/joss.03021](https://doi.org/10.21105/joss.03021)

*Wolf T, Debut L, Sanh L, et al. Transformers: State-of-the-Art Natural Language Processing. Proceedings of EMNLP 2020: System Demonstrations. Association for Computational Linguistics*, 38–45 (2020). [https://doi.org/10.18653/v1/2020.emnlp-demos.6](https://doi.org/10.18653/v1/2020.emnlp-demos.6)

*Zhang T, Kishore V, Wu F, Weinberger KQ, Artzi Y. BERTScore: Evaluating Text Generation with BERT. International Conference on Learning Representations (ICLR)* (2020). [https://arxiv.org/abs/1904.09675](https://arxiv.org/abs/1904.09675)


## Citation

If you use this repository, please cite:

```bibtex
@article{
  title   = {Measuring the Quality of AI-Generated Clinical Notes: A Systematic Review and Experimental Benchmark of Evaluation Methods},
  author  = {Alexandra Dahlberg, Tiila Käenniemi, Tiia Winther-Jensen, Olli Tapiola, Rami Luisto, Tuukka Puranen, Max Gordon, Enni Sanmark and Ville Vartiainen},
  year    = {2025},
  journal = {To appear}
}
```

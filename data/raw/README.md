# Sample document for the demo & evaluation

The evaluation set (`eval/eval_set.json`) and the sample demo are grounded in:

> K. Bonawitz, H. Eichner, W. Grieskamp, D. Huba, A. Ingerman, V. Ivanov, C. Kiddon,
> J. Konečný, S. Mazzocchi, H. B. McMahan, T. Van Overveldt, D. Petrou, D. Ramage,
> and J. Roselander. **"Towards Federated Learning at Scale: System Design."**
> Proceedings of MLSys, 2019. [arXiv:1902.01046](https://arxiv.org/abs/1902.01046)

The PDF is **not committed** to this repository (it is a third-party paper). To run
the demo and reproduce the evaluation, download it into this folder:

```bash
# from the repo root
curl -L -o data/raw/federated_learning.pdf https://arxiv.org/pdf/1902.01046
```

The `relevant_pages` in `eval/eval_set.json` refer to the page numbers of this exact PDF.
You can use any of your own PDFs instead — just write a matching `eval_set.json`.

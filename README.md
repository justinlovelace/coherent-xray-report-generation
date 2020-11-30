# coherent-xray-report-generation
This contains the code for the paper "Learning to Generate Clinically Coherent Chest X-Ray Reports" published in the Findings of EMNLP 2020. Unfortunately the dataset that we worked on is not publicly available, but we release our modeling code so that people can more directly build on or repurpose our methods.



If you find our work or our code helpful in your work then please cite our paper.

```
@inproceedings{lovelace-mortazavi-2020-learning,
    title = "Learning to Generate Clinically Coherent Chest {X}-Ray Reports",
    author = "Lovelace, Justin  and
      Mortazavi, Bobak",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.110",
    pages = "1235--1243",
    abstract = "Automated radiology report generation has the potential to reduce the time clinicians spend manually reviewing radiographs and streamline clinical care. However, past work has shown that typical abstractive methods tend to produce fluent, but clinically incorrect radiology reports. In this work, we develop a radiology report generation model utilizing the transformer architecture that produces superior reports as measured by both standard language generation and clinical coherence metrics compared to competitive baselines. We then develop a method to differentiably extract clinical information from generated reports and utilize this differentiability to fine-tune our model to produce more clinically coherent reports.",
}

```

### Acknowledgements
We built upon code released in the following repos:

- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
- https://github.com/fawazsammani/knowing-when-to-look-adaptive-attention
- https://github.com/wboag/cxr-baselines

Our implementation of the Gumbel Softmax trick was adapted from https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
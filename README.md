# A Three-Player Game for Rationalization
This repo contains the PyTorch implementation of the EMNLP 2019 paper [Rethinking Cooperative Rationalization: Introspective Extraction and Complement Control](https://people.csail.mit.edu/tommi/papers/YCZJ_EMNLP2019.pdf).  To make this repo neat and light-weight, we release the core code and data for the newly proposed single-aspect beer review dataset (i.e. the evaluation on the left of Table 4 in the paper) for the demo purpose.   If you are interested in reproducing the exact results for other datasets, please contact us, and we are very happy to provide the code and help.  

You can start with the following entry script.
```bash
run_beer_single_aspect_rationale_3players.py
```

**Data requirement:**
Please download the beer review data following the paper [Rationalizing Neural Predictions](https://arxiv.org/pdf/1606.04155.pdf), then put ```data/sec_name_dict.json``` to your data directory.

**Tested environment:**
Python 2.7.13, PyTorch: 0.3.0.post4  

If you find this work useful and use it in your research, please consider to cite our paper.

```
@inproceedings{yu2019rethinking,
  title={Rethinking Cooperative Rationalization: Introspective Extraction and
Complement Control},
  author={Yu, Mo and Chang, Shiyu and Zhang, Yang and Jaakkola, Tommi S},
  booktitle={Empirical Methods in Natural Language Processing},
  year={2019}
}
```

## Final Words
That's all for now and hope this repo is useful to your research.  For any questions, please create an issue or email gflfof@gmail.com, and we will get back to you as soon as possible.

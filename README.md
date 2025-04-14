# BENO with GPS
This is the repository for mini-project "GraphGPS for Boundary Encoding in MPNN-Based
Neural PDE Solvers" for the *Geometric Deep Learning* , based on the paper

[BENO:Boundary-embedded Neural Operator for Elliptic PDEs (ICLR 20244)](https://openreview.net/forum?id=ZZTkLDRmkg) ([arXiv](https://arxiv.org/abs/2401.09323) | [Tweet](https://twitter.com/tailin_wu/status/1747259448635367756)) by [Haixin Wang*](https://willdreamer.github.io/), [Jiaxin Li*](https://github.com/Jiaxinlia/Jiaxin.github.io), [Anubhav Dwivedi](https://dwivedi-anubhav.github.io/website/), [Kentaro Hara](https://aa.stanford.edu/people/ken-hara), [Tailin Wu](https://tailin.org/).

The mini-project explores the use of GPS transformer instead of vanilla transformer for boundary embedding. All training and tested are conducted using `train_colab.ipynb` on google colab using a T4. 

`transformer_gps.py` and `BE_MPNN_GPS.py` contain Python scripts for implementing BENO with GraphGPS for boundary encoding.




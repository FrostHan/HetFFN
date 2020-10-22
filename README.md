# HetFFN

Codes for paper "Lamina-specific neuronal properties promote robust, stable signal propagation in feedforward networks", NeurIPS 2020

### Dependences:

- Python==2.7.13 (Python 3 should also work)
- Brian2==2.1.1
- h5py==2.7.0
- scipy==0.19.1
- matplotlib==2.0.2


### To generate the data for Fig. 1(C) and Appendix Fig. A1

```
python Single_neuron_threshold.py
```


### To generate the data for Fig. 2 and Appendix Fig. A4

```
python Jeanne-Wilson-2015-DEsynapse.py
python Jeanne-Wilson-2015-DEsynapse-OUinput.py
```


### To generate the data for Fig. 3, 4 Appendix Fig. A2, A5

```
python Deep-FFN.py
python Deep-FFN-OUinput.py
```

### To generate the data for Appendix Fig. A3

```
python Deep-FFN-Inh.py
```

Data will be saved to ./data/ in .mat format.

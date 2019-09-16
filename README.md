#Domain Transfer in Dialogue Systems without Turn-Level Supervision
 
This is the code repository for our paper [Domain Transfer in Dialogue Systems without Turn-Level Supervision](url).

## Run it

##### Install dependencies
```
pip install -r requirements.txt
```

#####Run training script
```
python run.py
```
This script gives you plenty of options for arguments 
controlling training, and it's also where you specifiy what
data you'll work on. Please run `python run.py -h` for 
an overview of options. 

#####Evaluating 
If you just want to evaluate a trained model and produce 
predictions, you do:
 ```
python run.py --test
```
 
 
## Cite
 
If you use this code, please cite the paper. Bibtex:
```
@article{bingel2019domain,
  title={Domain Transfer in Dialogue Systems without Turn-Level Supervision},
  author={Bingel, Joachim and Petr\'en Bach Hansen, Victor and Gonzalez, Ana Valeria and Budzianowski, Pawe{\l}  and Augenstein, Isabelle and S{\o}gaard, Anders},
  journal={arXiv preprint arXiv:1909.[to-come]},
  year={2019}
}

```
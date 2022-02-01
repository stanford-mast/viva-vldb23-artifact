# VIVA

## Installation instructions
Clone this repository and fetch submodules with:
```
git clone --recursive https://github.com/faromero/viva.git
git submodule update --init --recursive
```
Install VIVA requirements:
```
sudo apt update  
sudo apt install python3-pip  
sudo apt install default-jre default-jdk ffmpeg  
cd viva  
python3 -m pip install -r requirements.txt
```
## Quickstart example
* Copy an mp4 video into the data/ directory
* Create the output/ directory in VIVA's root directory with: `mkdir output`
* Run `run_query.py`:
```
python3 run_query.py [-h] [--logging [LOGGING]] [--cache] [--ingestwarmup] [--selectivityfraction SELECTIVITYFRACTION]
                     [--selectivityrandom] [--costminmax {min,max}] [--f1thresh F1THRESH] [--opttarget {performance,cost,dollar}]
                     [--query {angrybernie,dunk,amsterdamdock,deepface,debug}] --canary CANARY [--logname LOGNAME]

optional arguments:
  -h, --help            show this help message and exit
  --logging [LOGGING], -l [LOGGING]
                        Do logging (optionally supply suffix for logfile name)
  --cache, -C           Enable caching and potential reuse of results
  --ingestwarmup, -w    Perform ingest (transcoding) warmup
  --selectivityfraction SELECTIVITYFRACTION, -s SELECTIVITYFRACTION
                        Fraction of frames to estimate selectivity over (Default: 0). 0 to disable estimating.
  --selectivityrandom, -r
                        Estimate selectivity by randomly choosing the fraction of frames. Not setting will do fixed rate.
  --costminmax {min,max}, -e {min,max}
                        Select plan based on min/max cost (Default: min)
  --f1thresh F1THRESH, -f F1THRESH
                        F1 threshold (Default: 0.8)
  --opttarget {performance,cost,dollar}, -o {performance,cost,dollar}
                        Plan optimization target (Default: performance)
  --query {angrybernie,dunk,amsterdamdock,deepface,debug}, -q {angrybernie,dunk,amsterdamdock,deepface,debug}
                        Query to run (Default: angrybernie)
  --canary CANARY       Canary input video to find database key.
  --logname LOGNAME
```

## Reference
If you find VIVA and/or Relational Hints useful in your work, please cite:
```
@inproceedings{relationalhints_pvldb23,
  title={Optimizing Video Analytics with Declarative Model Relationships},
  author={Francisco Romero and Johann Hauswald and Aditi Partap and Daniel Kang
          and Matei Zaharia and Christos Kozyrakis},
  year={2023},
  booktitle={PVLDB}
}
```

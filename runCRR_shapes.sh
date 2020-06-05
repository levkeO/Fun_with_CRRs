#!/bin/bash
python3 CRR_shape.py '../glass_KA21/T0.48/' T0.480_N10002_KA21.xyz 20 0.48 350
python3 CRR_shape.py '../glass_KA21/T0.49/' T0.490_N10002_KA21.xyz 30 0.49 512 
python3 CRR_shape.py '../glass_KA21/T0.5/' T0.500_N10002_KA21.xyz 17 0.5 512 
python3 CRR_shape.py ../glass_KA21/T0.52/ T0.52_N10002_NVT_step_1000LJ_startFrame300.xyz 170 0.52 696
python3 CRR_shape.py  ../glass_KA21/T0.55/ T0.55_N10002_NVT_step_100LJ_run2.xyz 324 0.55 1000
python3 CRR_shape.py ../glass_KA21/T0.6/ T0.6_N10002_NVT_step_10LJ_startFrame500.xyz 200 0.6 1000
python3 CRR_shape ../glass_KA21/T0.7/CRR/ T0.7_N10002_NVT_step_5LJ_startFrame500.xyz 25 0.7 1000
python3 CRR_shape.py ../glass_KA21/T0.8/CRR/ T0.8_N10002_NVT_step_1LJ_startFrame500.xyz 26 0.8 1000

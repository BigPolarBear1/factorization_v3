New personal best! Big update!!</br>
I have uploaded the new PoC files in the folder PoC-Files.</br>
To compile using cython:</br></br>
From the PoC-Files folder: python3 setup.py build_ext --inplace</br></br>
The below benchmarks assumes 8 CPU cores available (benchmarks ran from an ubuntu VM):</br></br>
To factor 100-bit:</br>
python3 run_qs.py -keysize 100 -base 500 </br> 
Running time (5 seconds)</br></br>

To factor 120-bit:</br>
python3 run_qs.py -keysize 120 -base 1000 </br> 
Running time (9 seconds)</br></br>

To factor 140-bit:</br>
python3 run_qs.py -keysize 140 -base 2000 </br> 
Running time (33 seconds)</br></br>

To factor 160-bit:</br>
python3 run_qs.py -keysize 160 -base 4000 </br> 
Running time (100 seconds)</br></br>

To factor 180-bit:</br>
python3 run_qs.py -keysize 160 -base 6000 </br> 
Running time (<to add> seconds)</br></br>

To factor 200-bit:</br>
python3 run_qs.py -keysize 160 -base 8000 </br> 
Running time (<to add> seconds)</br></br>

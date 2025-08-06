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
python3 run_qs.py -keysize 180 -base 6000 </br> 
Running time (400 seconds)</br></br>

To factor 200-bit:</br>
python3 run_qs.py -keysize 200 -base 8000 </br> 
Running time (800 seconds)</br></br>

---------------------------------------------------------------------------
Notes and to do: The uploaded will get to above 200-bit, or 60 digits. I borrowed some of the logic from SIQS and incorporate it into my own work. The assumed hard cap for what would be factorable on a single machine using quadratic sieve would be around 100-110 digits.
This gets us a little closer then v2. (that 110 digit cap assumes an algorithm that's highly optimized, written in a low level language using every trick from the book.. this one is currently unoptimized, using cython to generate shitty c-code without static typing.. for now.. so my hope is that once I account for all that, I can get past that).

Currently it is build using cython, but there are no optimizations such as static typing yet, so it is basically not doing anything yet for performance. 

Major things still left to do:</br>
-Experiment with lifting (since we have already figured out the math, we should check if it makes a difference)</br>
-Implement bottlenecks in C or atleast Cython (i.e add static typing) and use a lib for large numbers</br>
-Implement large prime variation (presumably doubles the performance)</br> 
-Implement a faster method for the linear algebra portion. Block Lanczos would be ideal. (lets see when this becomes an issue first)</br> 
-The factor_base (building iN map) is still a major bottleneck. Much of this could be saved to an offline database.. but I need to think how to speed it up some more.</br>
-Lots of small optimizations still to be done (i.e skip small primes for the sieve interval).. I should also think if there is a point in a negative sieve_interval (going the negative direction).</br> </br>  

I will begin addressing these things now. And also optimize whereever I can. If we can push past 110 digits on a laptop... then the fun begins. But until then... the doubt and agony remains.. but there is yet hope.

References from which I copied and modified some code (i.e the randomized modulus selection logic for the linear coefficient, which in their version is the divisors of the quadratic coefficient and the guassian elimination logic): https://stackoverflow.com/questions/79330304/optimizing-sieving-code-in-the-self-initializing-quadratic-sieve-for-pypy

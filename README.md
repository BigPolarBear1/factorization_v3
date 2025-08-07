UPDATE: Just added large prime variation.

New personal best! Big update!!</br>
I have uploaded the new PoC files in the folder PoC-Files.</br>
To compile using cython:</br></br>
From the PoC-Files folder: python3 setup.py build_ext --inplace</br></br>
The below benchmarks assumes 8 CPU cores available (benchmarks ran from an ubuntu VM):</br></br>
To factor 100-bit:</br>
python3 run_qs.py -keysize 100 -base 500 </br> 
Running time (4 seconds)</br></br>

To factor 120-bit:</br>
python3 run_qs.py -keysize 120 -base 1000 </br> 
Running time (4 seconds)</br></br>

To factor 140-bit:</br>
python3 run_qs.py -keysize 140 -base 1000 </br> 
Running time (8 seconds)</br></br>

To factor 160-bit:</br>
python3 run_qs.py -keysize 160 -base 2000 </br> 
Running time (33 seconds)</br></br>

To factor 180-bit:</br>
python3 run_qs.py -keysize 180 -base 4000 </br> 
Running time (109 seconds)</br></br>

To factor 200-bit:</br>
python3 run_qs.py -keysize 200 -base 6000 </br> 
Running time (450 seconds)</br></br>

To factor 220-bit:</br>
python3 run_qs.py -keysize 220 -base 10000 </br> 
Running time (2500 seconds)</br></br>

References

Reference #1: https://stackoverflow.com/questions/79330304/optimizing-sieving-code-in-the-self-initializing-quadratic-sieve-for-pypy (I copied a bunch of code from this and modified it to fit my own work... mainly as related to the randomized modulus selection for linear coefficients (which they do for the divisors of their quadratic coefficients). They use pypy and the solution proposes to write part in c++, I may actually move to that approach as well rather then using cything. Just start porting everything to c++)

---------------------------------------------------------------------------
Update: I have just added the large prime variation and minimum prime threshold for sieving. This roughly doubled the performance.

Next I need to investigate if sieving in the negative direction is going to help in my specific usecase (since it looks quite different from regular SIQS since we use linear coefficients).
After that I will also need to investigate lifting. Since it can be done reasonably fast and I've already figured out the math for it and have most of the code in place already.

After that I will need to actually do low level optimizations, it's currently being cythonized, but without static typing, it's not adding much (if anything) to performance compared to regular python.
Plus I will also need to implement using a large number library since that will help too.

And lastly there is a bunch fo small things I need to look at to get performance increases, and just general code flow stuff.
Once all of that is done, I think I will finish factorization_v3 and move on to factorization_v4 where I will continue with pure number theory for a while until I get a breakthrough that allows me to improve this work.

Oh and ofcourse once I hit a point where the linear algebra section becomes a bottleneck, implementing block lanczos would be great.

I'll push updates regulary to address all these things. I want to atleast outperform msieve.. that is my goal for v3.


Update: I am depressed. I was doing some interesting stuff in with this research about a month or two months ago. I just really want to backtrack and continue from there. But on the other hand, if my research is any better then what's already out there, then this should improve msieve's SIQS implementation... even if it is by a tiny amount. Hence I should go the extra mile here and make it happen. Atleast it would give me some kind of proof I can make noise about... post on forums, etc. Without that proof, people will just laugh at me like they always do. Truely a shit industry. Just like microsoft. Easier for their ego to downplay everything I do in my life. I don't understand why people who try half as hard, with half the talent can earn six figure salaries doing fuck all and I can't even get a job interview lol.



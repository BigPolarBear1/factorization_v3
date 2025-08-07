UPDATE: Just added large prime variation.

Run this with pypy3 for best performance.
I will move most to pure c++ in the future.

To install pypy3: </br></br>

sudo add-apt-repository ppa:pypy/ppa</br>
sudo apt update</br>
sudo apt install pypy3</br></br>

The below benchmarks assumes 8 CPU cores available (benchmarks ran from an ubuntu VM) and have been tested on a machine with 64gb of ram.. lower worker count if ram is an issue: </br></br>
To factor 100-bit:</br>
pypy3 QSv3_050.py -keysize 100 -base 500 </br> 
Running time (4 seconds)</br></br>

To factor 120-bit:</br>
pypy3 QSv3_050.py -keysize 120 -base 1000 </br> 
Running time (4 seconds)</br></br>

To factor 140-bit:</br>
pypy3 QSv3_050.py -keysize 140 -base 1000 </br> 
Running time (5 seconds)</br></br>

To factor 160-bit:</br>
pypy3 QSv3_050.py -keysize 160 -base 2000 </br> 
Running time (17 seconds)</br></br>

To factor 180-bit:</br>
pypy3 QSv3_050.py -keysize 180 -base 4000 </br> 
Running time (90 seconds)</br></br>

To factor 200-bit:</br>
pypy3 QSv3_050.py -keysize 200 -base 6000 </br> 
Running time (450 seconds)</br></br>

To factor 220-bit:</br>
pypy3 QSv3_050.py -keysize 220 -base 10000 </br> 
Running time (2500 seconds)</br></br>

References

Reference #1: https://stackoverflow.com/questions/79330304/optimizing-sieving-code-in-the-self-initializing-quadratic-sieve-for-pypy (I copied a bunch of code from this and modified it to fit my own work... mainly as related to the randomized modulus selection for linear coefficients (which they do for the divisors of their quadratic coefficients). They use pypy and the solution proposes to write part in c++, I may actually move to that approach as well rather then using cython. Just start porting everything to c++ until just the outer-most logic is still in python)

---------------------------------------------------------------------------
Update: I have just added the large prime variation and minimum prime threshold for sieving. This roughly doubled the performance.

Next I need to investigate if sieving in the negative direction is going to help in my specific usecase (since it looks quite different from regular SIQS since we use linear coefficients).
After that I will also need to investigate lifting. Since it can be done reasonably fast and I've already figured out the math for it and have most of the code in place already.

After that I will need to start moving critical parts of the code to c++. I hope to eventually move 90% to c++ as pypy eats up a lot of ram and the only reason I'm not doing c++ from scratch is becaues I am depressed and in my depressed state it feels like an herculean task. The performance is still quite low right now. It's not optimized for pypy either as the PoC on stackoverflow is. But I don't want to waste time on that and just jump straight into c++.

My primary goal is to beat msieve. I'm not happy about this research direction at all. Looking back, I was doing some interesting stuff about a month or two ago and I really want to backtrack and continue doing research in that direction in addition to doing more pure number theoretical research. But if any tiny piece in my work (i.e using linear coefficients) is any different from what's already out there... then this should in theory outperform msieve... hence for my own sanity, I must go the extra mile here and outperform msieve. And the moment I succeed, I have something I can share on the internet and make noise about. It's best to back up claims with hard proof, otherwise it's a losing battle.


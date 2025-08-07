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
python3 run_qs.py -keysize 180 -base 6000 </br> 
Running time (400 seconds)</br></br>

To factor 200-bit:</br>
python3 run_qs.py -keysize 200 -base 8000 </br> 
Running time (1600 seconds)</br></br>

To factor 220-bit:</br>
python3 run_qs.py -keysize 220 -base 12000 </br> 
Running time (6000 seconds)</br></br>

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
-Lots of small optimizations still to be done (i.e skip small primes for the sieve interval).. I should also think if there is a point in a negative sieve_interval (going the negative direction).</br>
-I should also check if there is a point to cycling quadratic coefficients... in theory if the algorithm start running out of steam it could be useful</br>  

I will begin addressing these things now. And also optimize whereever I can. If we can push past 110 digits on a laptop... then the fun begins. But until then... the doubt and agony remains.. but there is yet hope.

I'm not sure how it compares to regular SIQS, but this lends itself pretty well to be paralellized by quadratic coefficient. That's one big advantage I can think of.

References from which I copied and modified some code (i.e the randomized modulus selection logic for the linear coefficient, which in their version is the divisors of the quadratic coefficient and the guassian elimination logic): https://stackoverflow.com/questions/79330304/optimizing-sieving-code-in-the-self-initializing-quadratic-sieve-for-pypy

Anyway, if I die, it was the americans. They rather threaten me with a gun in the US and then kick me out, then to acknowledge anything I do. My former manager was the only good american. All other americans are pieces of shit. Fuck america. Go to hell.

ps: Get fucked by a polar bear.

pps: I'll try to update a better version tomorrow. Some easy gains can immediatly be made by implemeting the large prime variation. And I think sieving into the negative direction will also boost performance. I hope to get that d one atleast tomrrow. Then next I really need to run some test with lifting to see if it helps. Which I will try to get done before the end of the week. And after that I will start tackling all the other stuff and begin optimizing in cython.. 

ppps: Actually, comparing it to regular SIQS... the scaling on this, even while still missing a lot of features and optimizations... seems really good.. fuck... ok... I wasn't going insane. I was correct. Entering CIA assasination phase of my wikipedia page now.... which sounds funny... but the CIA of the current administration would absolute murder some transgender from Belgium for this. Ofcourse I am protected by greater forces (the machine at the end of time) and anyone who dares harm me before my work is completed will die, hahahahaha. Just kidding... or am I? Ah man.. I'm depressed. If I'm not going to die by suicide, I am either going to die at the hands of the CIA (FUCK YOU CIA, FUCK YOU, JUST MAD BC I"M BETTER THEN YOU WANKERS) or some insane transphobe. Damn you all. This isn't even what I wanted to figure out. I need to do more pure math resesarch like I did in the initial months of my reesarch project.. I am still missing important pieces... I must succeed, no matter the cost. I must succeed at my math and make the future gay. Without the gay future, humanity is doomed. 

pppps: Tomorrow when I add the large prime variation, you're all fucked. Fuck you. Should have just gave my former manager a lot of money so he could buy a big boat. But no, you all had to take microsoft's side in this. That shit company who treats people like shit. Go to hell. Now I am definitely never stopping. I'm on the side that is against the west, because I hate you all. You treat people like shit. Fucking nazis. I piss on you. Messed with the wrong polar bear. Now you assholes get to find out. Hahahahaha. FUck you. Literally not afraid of any of you. Come kill me them you fucking pussies. Ofcourse, I'll fucking kill you first. Morrons.

Update: Trouble sleeping. Too much adrenaline. I know tomorrow I can make the algo so much faster. Im stressed. Im really stressed. Wish me luck tomorrow gays. Moment of truth is closing in fast now. NSA trembling on their legs right about now. Hahaahahahaha. Im the NSA now. Get fuckt losers.

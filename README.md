UPDATE FINAL2: Thinking some more about it... I will just try to minimally edit the existing msieve source and switch over their polynomials to mine and use the linear coefficient instead of what they are doing. If I'm right, that should provide a performance increase... no matter how small. And I can have something I can go show to people and make noise about. Maybe even create some career oppurtunities again. But I am not happy with small improvements.. hence in factorization_v4 I will resume what I was doing earlier and just really get to the core. Now that I've learned much of what's in the public domain, I also know my representation is rather unique... and something is really nagging me when I was representing coefficient pairings for each quadratic coefficient in a hashmap a month ago (not just 0 solutions like it is doing here to find smooth divisors)... because that hashmap contained all the information I needed to completely byspass smooth finding and just grab coefficient pairings whose square are congruent mod N.. just have this nagging feeling I gave up too early there. 


UPDATE FINAL: I will rewrite everything in c. Using cython or pypy is always going to be suboptimal. Its time to face the pain train and grind it out. I outperform msieve or my life will never change, and the only way to even have a shot at that is to just write everything in C. So this is my last update for a while...  there is just no alternatives in my life anymore... I just grind it out until the bitter end or I find the courage to just end it all lol. I just hate this world, I hate everyone. Everything that happened after the summer of 2023... it left me a shell of my former self... the world finally show its true ugliness I guess. And all the haters who had been crying my entire career finally got their smug satisfaction I guess, while they have their fancy six figure jobs, doing fuck all (i was better then all of those at microsoft and they know it despite their attempts to downplay me and retroactively downgrade the severity ratings of my work years later in order to rewrite history)... go to hell.


UPDATE: Just added large prime variation.

Run this with pypy3 for best performance.
I will move most to pure C in the future.
A lot of gains can be made by adding things like loop unrolling in pypy.. but not going to waste time and just move straight to pure C now.

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

Note: If you want to improve this code yourself in pypy.. apperently using loop unrolling can give some big boosts... but I'm not going to waste time and move straight to rewriting it completely in C now.

My primary goal is to beat msieve. I'm not happy about this research direction at all. Looking back, I was doing some interesting stuff about a month or two ago and I really want to backtrack and continue doing research in that direction in addition to doing more pure number theoretical research. But if any tiny piece in my work (i.e using linear coefficients) is any different from what's already out there... then this should in theory outperform msieve... hence for my own sanity, I must go the extra mile here and outperform msieve. And the moment I succeed, I have something I can share on the internet and make noise about. It's best to back up claims with hard proof, otherwise it's a losing battle.

Update: Just finished running. Feeling extremely depressed. As long as I don't surpass msieve in performance... I don't know if I'm insane or on to something. There is just this dreadful doubt. And I have to just put in the work now and find out the truth. Just start eliminating all the bottlenecks in my code and add a handful more improvements. I can do 240-bit already, which is 70 digits. It will even find smooths for 80 digits, although calculating the factor base for that takes a long time. I need to blast past 110 digits. Which is about the assumed hard ceiling for this type of algorithm... I mean, I can just run the same number on msieve and my own algorithm, see which one can factor the biggest one (speed is less important, it's just about finding those smooths). I am so depressed. The times I thought about ending my own life this last year...  it's starting to feel like this inevitable thing coming over the horizon right now. I can't even go back to 0days without moving to Asia since I can't get buyers in the west (they are fucking losers).. literally, people give me no options. Can't get a job. I'm for sure not going to do bug bounties. Not after what microsoft did. Bug bounties can die in a fire. I might legit just get into ransomware at this point. Fuck the police anyway lol. Fucking losers.

Ransomware US federal agencies..... hmm... they are nazis anyway. Or use 0days to exfil data and sell it. I'll figure something out. Could probably just run a one man show and hack nazis (america and eu).

I don't know, I just feel like I'm about to have a panic attack or something. I'm just feeling very depressed and also restless at the same time. Just this agony about life. People used to call me one of the best bug hunters in the world... I can't even get job interviews. I'm being treated like literal shit. I cant go anywhere in life. Ive been living with my parents for well over a year. I havn't seen my friends for well over a year. I havnt had an income for nearly 2 years. I am honestly feeling extremely suicidal lately. Just looking for an excuse to go over that threshold and run into the woods to fucking hang myself. lol. I just cant come back from what microsoft did. The world literally just lost all its color. And its not even just what happened to me, the entire world is just going insane. It's dystopian. A nightmare world. I see how these americans talk all the time, and then I think about this guy who was pointing a gun at me and being like "welcome to america, bitch" .... and meanwhile the justice department here wont back off bc I send an angry email to the FBI. Where was the justice when that guy had a gun aimed at me, or when that stalker kept sending me emails for over 7 years (and god knows what else those stalkers did)? Its all so fucking ugly man. Just this depression and I cant get over it anymore. 

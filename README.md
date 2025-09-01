Note: I know I am right. I know people know I am right. I know how you people treat me, despite being right. I do not care about the reasons. I will not forgive you people for stealing more then a year from my life where I had to struggle like this. I especially wont forgive belgium. I would renounce my citizenship and move elsewhere for this. I cannot ever forgive this. I've been broke for a year, I can't find work, I can't find 0day buyers, I can't see my friends anymore and I'm stuck, there is no dignity in living like this. A human life is very short, and you have wasted a good chunk of mine. There absolutely won't be forgiveness. We're past that now. And if I get the oppurtunity to work in Asia, I will go and there is absolutely nothing that can be changed about that. Because what you people are doing, I have no words for it, this is betrayal. Betrayal to the long career I had, helping find security issues in software like OpenSsl and other very important components.. you better hope this was my last discovery because next time I will remember who treated me like shit and who didn't. Remember that. The only way I may even remotely change my mind is if Microsoft publically apologizes to my former manager and pays him many millions. Without that, forget it, you'll have to kill me I guess.

Disclaimer about v3: My main goal for this version is to outperform standard SIQS by being able to quickly precalculate what is in SIQS the roots of the polynomials (but here are coefficients mod m). In theory being able to solve that bottleneck should allow us to outperform standard SIQS. And once I am able to proof this with a PoC, I hope to make some noise about it and find a way to get funded for v4, and I don't care if that means moving to Asia. I'll do whatever it takes to work on this. Sure as hell have not been gettig respect here in the west.

Update 31 August 2025

Uploaded QSv3_simd.pyx 

To build (from the PoC_Files folder): python3 setup.py build_ext --inplace</br></br>
To run: (note the actual factor base per quadratic coefficient will be about +/- half of the -base parameter)

140 bit: python3 run_qs.py -base 1000 -keysize 140 -debug 0 -lin_size 100_000 -quad_size -1 (2 seconds)</br>
160 bit: python3 run_qs.py -base 2000 -keysize 160 -debug 0 -lin_size 100_000 -quad_size -1 (8 seconds)</br>
180 bit: python3 run_qs.py -base 4000 -keysize 180 -debug 0 -lin_size 100_000 -quad_size -1 (51 seconds)</br>
200 bit: python3 run_qs.py -base 6000 -keysize 200 -debug 0 -lin_size 100_000 -quad_size -1 (196 seconds)</br>
220 bit: python3 run_qs.py -base 10000 -keysize 220 -debug 0 -lin_size 100_000 -quad_size -1 (770 seconds)</br>
240 bit: python3 run_qs.py -base 20000 -keysize 240 -debug 0 -lin_size 100_000 -quad_size -1 (2 hours)</br> (+/- 73 digits)
The PoC is still very unoptimized... lets see how far we can push it. Msieve will really struggle around 350-bit (a highly optimized SIQS PoC), so we need to push beyond that to be succesful.

Currently this only works with 1 worker. I have gutted the worker support to better be able to profile the code. Will fix later.

OMG. So PoC is now bottlenecking at calculating linear congruences. But I was doing some reading about number theory. The way you solve this is with what I call "partial results" in my paper. But the way I was doing it was over complicated. You just have to calculated one inverse of the total modulus to achieve the same. So we can calculate these "partial results" really fast. Ok ok, fuck. I'm stupid. It's these obvious things that I suspected were possible but I never even took 10 minutes of my life to just check it. My brain is highly illogical at times.

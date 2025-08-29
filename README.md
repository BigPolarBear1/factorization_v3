Note: I know I am right. I know people know I am right. I know how you people treat me, despite being right. I do not care about the reasons. I will not forgive you people for stealing more then a year from my life where I had to struggle like this. I especially wont forgive belgium. I would renounce my citizenship and move elsewhere for this. I cannot ever forgive this. I've been broke for a year, I can't find work, I can't find 0day buyers, I can't see my friends anymore and I'm stuck, there is no dignity in living like this. A human life is very short, and you have wasted a good chunk of mine. There absolutely won't be forgiveness. We're past that now. And if I get the oppurtunity to work in Asia, I will go and there is absolutely nothing that can be changed about that. Because what you people are doing, I have no words for it, this is betrayal. Betrayal to the long career I had, helping find security issues in software like OpenSsl and other very important components.. you better hope this was my last discovery because next time I will remember who treated me like shit and who didn't. Remember that. The only way I may even remotely change my mind is if Microsoft publically apologizes to my former manager and pays him many millions. Without that, forget it, you'll have to kill me I guess.

Update 28 August 2025

Uploaded QSv3_simd.pyx 

To build (from the PoC_Files folder): python3 setup.py build_ext --inplace</br>
To run: python3 run_qs.py -base 2000 -keysize 150 -debug 1 -lin_size 100_000 -quad_size -10  (note the actual factor base per quadratic coefficient will be about +/- half of the -base parameter)

note: Only one worker at a time works for now... I need to rework that part.

To use the old PoC (that one will easily factor above 200 bit using pypy3):

pypy3 QSv3_050.py -base 6000 -keysize 200

Added another round of refactoring. The approach I've settled now on is to get rid of as much code as possible. Since this seems to be the most important thing and then precompute as much as possible also.
I'll do some more big changes tomorrow. 
Then the sieve row and simd function... we need to chunk that part, so it doesn't have to construct a huge sieve row each time... and we can precompute the chunks.
And when all of that is done... the real optimizing begins... we need to use memory views, get rid of all the python abstraction, debug the generated C code. 
I want to move all of the heavy computational burden to before the main loop... I also need to optimize constructing the iN map at the start of the algo... that needs to use numpy arrays to save memory.
And I also need to use proper threading, since worker support doens't work right now. And a bunch of other things...  anyway, minimizing the code and then fixing array indexing seems to be the big ticket work item I need to address.

I kind of got distracted for a little bit unsure on what the best appraoch is... but from experimenting, most important is reducing code. Using contiguous memory access without all the python abstraction or numpy strided access. When all of that is done, maybe we can get some more juice using SIMD. But it's not the most important tool. Unless you have some specialized SIMD hardware or something. 

And where my work shines is that we can switch polynomials without recomputing roots. I just got to optimize my shit code. The only thing holding me back is my shit code. Nothing else. Anyway, got it now, I know what to do.. time to grind it out. 

Update: Ergh, just did the math. I'm a fucking idiot. If you have a coefficient, you just subtract the modulus to do negative sieving. Simple as that. I thought it would just loop back and generate the same smooths twice, but it doens't. I should not make these stupid assumptions in my head without running the numbers. Let me add that first thing tomorrow.... that will double the smooths... fucking hell. Lol. I do wonder though... i.e if we have 66 as coefficient and 148 ... then 148^2 - 4\*4387 will yield a smaller number then 66^2+4\*4387 ... so it may be good to swap the coefficients I'm using as a test.

Cant sleep. Brain is overactive. I should be able to make fast progress now the next few days since ive narrowed down what I need to do and what will result in the biggest performance gains. 

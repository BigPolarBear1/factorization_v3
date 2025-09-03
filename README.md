Note: I know I am right. I know people know I am right. I know how you people treat me, despite being right. I do not care about the reasons. I will not forgive you people for stealing more then a year from my life where I had to struggle like this. I especially wont forgive belgium. I would renounce my citizenship and move elsewhere for this. I cannot ever forgive this. I've been broke for a year, I can't find work, I can't find 0day buyers, I can't see my friends anymore and I'm stuck, there is no dignity in living like this. A human life is very short, and you have wasted a good chunk of mine. There absolutely won't be forgiveness. We're past that now. And if I get the oppurtunity to work in Asia, I will go and there is absolutely nothing that can be changed about that. Because what you people are doing, I have no words for it, this is betrayal. Betrayal to the long career I had, helping find security issues in software like OpenSsl and other very important components.. you better hope this was my last discovery because next time I will remember who treated me like shit and who didn't. Remember that. The only way I may even remotely change my mind is if Microsoft publically apologizes to my former manager and pays him many millions. Without that, forget it, you'll have to kill me I guess.

Disclaimer about v3: My main goal for this version is to outperform standard SIQS by being able to quickly precalculate what is in SIQS the roots of the polynomials (but here are coefficients mod m). In theory being able to solve that bottleneck should allow us to outperform standard SIQS. And once I am able to proof this with a PoC, I hope to make some noise about it and find a way to get funded for v4, and I don't care if that means moving to Asia. I'll do whatever it takes to work on this. Sure as hell have not been gettig respect here in the west.

Update 1 September 2025

Uploaded QSv3_simd.pyx 

To build (from the PoC_Files folder): python3 setup.py build_ext --inplace</br></br>
To run: (note the actual factor base per quadratic coefficient will be about +/- half of the -base parameter and currently worker support is also not working)

140 bit: python3 run_qs.py -keysize 140 -base 1000 -debug 1 -lin_size 100_000  -quad_size 1 (2 seconds)    </br>
160 bit: python3 run_qs.py -keysize 160 -base 1000 -debug 1 -lin_size 100_000  -quad_size 1 (8 seconds)    </br>
180 bit: python3 run_qs.py -keysize 180 -base 2000 -debug 1 -lin_size 100_000  -quad_size 1 (16 seconds)   </br>
200 bit: python3 run_qs.py -keysize 200 -base 6000 -debug 1 -lin_size 100_000  -quad_size 1 (100 seconds) </br>

The PoC is still very unoptimized... lets see how far we can push it. Msieve will really struggle around 350-bit (a highly optimized SIQS PoC), so we need to push beyond that to be succesful.

Alright, solved the linear congruence bottleneck.

To do:

It is now bottlenecking in temp_split() and construct_interval_2(). Which should not be bottlenecks. So I need to investigate why this is happening. I'm guessing slow python indexing.
Secondly, I need to go over all linear co permutations instead of doing just one linear co, since there is a trick to do this very quickly. We may need to remove the negative sieving to avoid duplicate coefficients, but overall it should yield a speed boost.
Thirdly, the precomputing of the factor_base, I know how to makes this much much much faster. But since it doesn't affect the main loop of the algorithm I'm keeping that optimization for the end.

We're getting very close ot the performance of optimized C scripts that are public, atleast below 200 bits. And we still have an awful lot of optimizing left to do and right now, it's still bottle necking MASSIVELY in places that it shouldn't, so a lot of speed gains should still be achievable. I am feeling very optimistic about this. We're nearly there now. Few more days. And a big problem is, as we go up in bit size, that bottlenecking in those functions becomes really severe.. so I am fairly sure it is happening due to slow indexing in factor_base related lists. So while we have gained speed at the <200 bits... we won't see much improvement until we address those bottlenecks. Let me go for a run. I'll do some proper profiling to max sure it is an indexing issue and tomorrow upload an improved PoC.

For 220 bit with -base 10000 we get this:</br>
Seconds</br>
225.76 QSv3_simd.pyx:598(temp_split) note: This shouldn't happen </br>
154.445 QSv3_simd.pyx:282(launch)  note: Ignore this one, this is due to building the factor base. We'll fix that later. </br>
102.659 QSv3_simd.pyx:625(construct_interval_2) note: This shouldn't happen </br>
52.506 QSv3_simd.pyx:615(miniloop_non_simd) note: not happy about this, but if we must it can be fixed with SIMD </br>
45.047 QSv3_simd.pyx:139(modinv) note: we can precompute these if we must </br>
41.670 QSv3_simd.pyx:734(generate_modulus) note: We'll rework this plus iterating lin co's mod m will nullify this bottleneck </br>
etc

   So there is definitely something happening that shouldn't be happening. I'm almost 100% sure it is indexing into python lists that grow in size as the factor base goes up. Because if we for exmple factor a 160-bit number, we can literally half the time it takes by using -base 1000 instead of -base 2000. Which doesn't make any sense.

Update: Back from running. Let me shower and get food. After that I'll address the bottlenecking in those two functions. Just make all the lists numpy arrays in those two functions and create typed memory views or whatever. Then hopefully we can grind away at 90 digit numbers within a day. And then we'll see whats left to do... I'm hoping I can just keep going round and round and get rid of bottlenecks until I outperform msieve. The next few days will tell...

Update: Bah, wasting time messing with numpy. If I convert lists to numpy arrays, I actually get a slow down. I'm trying to figure out why. I should probably follow this to the letter: https://cython-docs2.readthedocs.io/en/latest/src/tutorial/numpy.html
And if that doesn't work I'll swap everything to the python arrays module. I should also probably manually add a bunch of overflow check and whatnot... because working with very large numbers, if anywhere in the code some hidden type conversions happen that shouldn't happen.. we get screwed over. Ergh. I hope I resolve this shit today.

Update: Blah, after wasting an entire day, I get it now. Got to check the generated C code and make sure there arn't type conversion all over the place in our inner loops.

Update: I'll try to upload a better version tomorrow. Kind of slowly understanding this cython stuff now... and learned the hard way you cant just make everything a memory view. Sometimes that overhead is not worth it. So for tomorrow I'll iterate linear coefficients.. we'll construct those sieve intervals in one go. Then I'm also going to add support for a sqldatabase for writing tonelli results too while building the factor base (the biggest bottleneck in building the factor base.. but something that can be trivially saved to disk for even huge factor bases and reused independent of N.. since we're calculating it for all quadratic residues mod p). And then just stuff like adding multi threading and a big number library. Thats another problem, right now we have to use native python int to support big ints.. and that also prevents us from statically typing a lot of stuff, since it will force type conversions. Even if we take the modulus of a native python int and we know the modulus is below 64 bit we can't type it to long long without forcing overhead from type conversions. It sucks but it is what it is.

I think once everything is optimized to the maximum and I have a better understanding of this cython stuff, what works and what doesn't... I really want to explore this idea of 2d sieving again too. I think it's interesting, but it needs to be implemented correctly. Hmm.

I really need to also try out p-adic lifting, bc we can calculate it quickly and it allows us to find more smooths with a smaller factor base. Which is really beneficial. Ah man, so much work so little time. Mentally I am really at the end of my rope.

Update: Bad day. Incredibly depressed. Waking up with suicidal thoughts again. It's the inevitable outcome slowly creeping over the horizon. What else do I do, cyberrime? I should get iterating linear coefficients done today. There's also a trick to quickly calculate the distance in the sieve interval. I wonder if I can use it to make my 2d interval idea work too... I just got to keep grinding a litte longer, find the strength. I also been experience these worsening issues with my gut for the last few months, feel like I'm dying mentally and physically. Fuck it. 

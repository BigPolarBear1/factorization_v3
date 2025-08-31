Note: I know I am right. I know people know I am right. I know how you people treat me, despite being right. I do not care about the reasons. I will not forgive you people for stealing more then a year from my life where I had to struggle like this. I especially wont forgive belgium. I would renounce my citizenship and move elsewhere for this. I cannot ever forgive this. I've been broke for a year, I can't find work, I can't find 0day buyers, I can't see my friends anymore and I'm stuck, there is no dignity in living like this. A human life is very short, and you have wasted a good chunk of mine. There absolutely won't be forgiveness. We're past that now. And if I get the oppurtunity to work in Asia, I will go and there is absolutely nothing that can be changed about that. Because what you people are doing, I have no words for it, this is betrayal. Betrayal to the long career I had, helping find security issues in software like OpenSsl and other very important components.. you better hope this was my last discovery because next time I will remember who treated me like shit and who didn't. Remember that. The only way I may even remotely change my mind is if Microsoft publically apologizes to my former manager and pays him many millions. Without that, forget it, you'll have to kill me I guess.

Update 31 August 2025

Uploaded QSv3_simd.pyx 

To build (from the PoC_Files folder): python3 setup.py build_ext --inplace</br>
To run: (note the actual factor base per quadratic coefficient will be about +/- half of the -base parameter)

140 bit: python3 run_qs.py -base 1000 -keysize 140 -debug 0 -lin_size 100_000 -quad_size -1 (3 seconds)</br>
160 bit: python3 run_qs.py -base 2000 -keysize 160 -debug 0 -lin_size 100_000 -quad_size -1 (20 seconds)</br>
180 bit: python3 run_qs.py -base 4000 -keysize 180 -debug 0 -lin_size 100_000 -quad_size -1 (51 seconds)</br>
200 bit: python3 run_qs.py -base 6000 -keysize 200 -debug 0 -lin_size 100_000 -quad_size -1 (232 seconds)</br>

Currently this only works with 1 worker. I have gutted the worker support to better be able to profile the code. Will fix later.

The more I started thinking about SIMD, the more I realized how wrong my approach was. If its only 256-bit writes, you only get an advantage if the data you are writing is spaced very close together (so you're not just writing a bunch of zeroes).
I removed it, and it sped up everything a lot. I need to actually do some reading on how to implement SIMD for sieve intervals. That seems to be an entirely independent research topic.

Anyway, by far the biggest bottleneck is now the linear congruences. But there is a well documented trick to massively speed that up. So I'll implement that next. Then we should be near the final shape of the algorithm.
What remains after that will be fixing indexing so we dont call into get_item() due to python abstractions. And adding static typing everywhere.
Once I'm happy with the mainloop, I'll shift my efforts to the precomputational part (I have barely touched that part so far, so there's lots of room for improvement there), since that is the strength of my findings, being able to shift all the burden there. Also wonder if I can apply that linear congruence trick to computing the iN map.. that would be amazing.. since that linear congruence is the biggest bottleneck there. Then there is no limit the the size of factor bases we can precompute... and then I would definitely feel vindicated about my work. In addition it needs to be placed in contiguous memory to save RAM.

And eventually I'll check about doing SIMD too. 

I'll solve the linear congruence bottleneck on Monday, its actually the biggest bottleneck now but there's a very well documented trick for it using constants... I'm hoping that not having to recalculate roots for my polynomials will allow me to overtake msieve's performance.. but I have to do everything right.. cant just do stupid shit like I was doing with the SIMD stuff anyway going to run 50k now. 

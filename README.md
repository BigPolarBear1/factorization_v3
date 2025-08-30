Note: I know I am right. I know people know I am right. I know how you people treat me, despite being right. I do not care about the reasons. I will not forgive you people for stealing more then a year from my life where I had to struggle like this. I especially wont forgive belgium. I would renounce my citizenship and move elsewhere for this. I cannot ever forgive this. I've been broke for a year, I can't find work, I can't find 0day buyers, I can't see my friends anymore and I'm stuck, there is no dignity in living like this. A human life is very short, and you have wasted a good chunk of mine. There absolutely won't be forgiveness. We're past that now. And if I get the oppurtunity to work in Asia, I will go and there is absolutely nothing that can be changed about that. Because what you people are doing, I have no words for it, this is betrayal. Betrayal to the long career I had, helping find security issues in software like OpenSsl and other very important components.. you better hope this was my last discovery because next time I will remember who treated me like shit and who didn't. Remember that. The only way I may even remotely change my mind is if Microsoft publically apologizes to my former manager and pays him many millions. Without that, forget it, you'll have to kill me I guess.

Update 30 August 2025

Uploaded QSv3_simd.pyx 

To build (from the PoC_Files folder): python3 setup.py build_ext --inplace</br>
To run: python3 run_qs.py -base 2000 -keysize 150 -debug 0 -lin_size 1 -quad_size -1 </br> (note the actual factor base per quadratic coefficient will be about +/- half of the -base parameter)

note: Only one worker at a time works for now... I need to rework that part.

Alright, I have gutted the worker support, so I can better profile the code. about 40 seconds to factor 150 bit. Buuuut, nearly all of that time in spent in construct_interval_simd().
Sadly cython doesn't support line-by-line profiling. But I'm fairly sure it is the python indexing causing this. There is something causing a massive slow down in that function and once that is fixed 150-bit should take about a second.
Because the total times that function gets called, vs the time we spent in the function doesn't make sense, something is definitely broken there.

I'm also removing the old pypy3 PoC since this one is coming close now in terms of performance... definitely once I figured out what is bottlenecking that function.

Anyway, I'm running 50k tomorrow and I should probably take a break for now to prepare and get some good sleep. I'll figure it out on monday. I might try to simulate line by line profiling by just making a bunch of smaller functions inside construct_interval_simd to isolate the root cause. And I should also look at thte C code, there may be some weird type conversion shit going on added extra overhead.

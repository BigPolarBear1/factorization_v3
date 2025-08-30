Note: I know I am right. I know people know I am right. I know how you people treat me, despite being right. I do not care about the reasons. I will not forgive you people for stealing more then a year from my life where I had to struggle like this. I especially wont forgive belgium. I would renounce my citizenship and move elsewhere for this. I cannot ever forgive this. I've been broke for a year, I can't find work, I can't find 0day buyers, I can't see my friends anymore and I'm stuck, there is no dignity in living like this. A human life is very short, and you have wasted a good chunk of mine. There absolutely won't be forgiveness. We're past that now. And if I get the oppurtunity to work in Asia, I will go and there is absolutely nothing that can be changed about that. Because what you people are doing, I have no words for it, this is betrayal. Betrayal to the long career I had, helping find security issues in software like OpenSsl and other very important components.. you better hope this was my last discovery because next time I will remember who treated me like shit and who didn't. Remember that. The only way I may even remotely change my mind is if Microsoft publically apologizes to my former manager and pays him many millions. Without that, forget it, you'll have to kill me I guess.

Update 29 August 2025

Uploaded QSv3_simd.pyx 

To build (from the PoC_Files folder): python3 setup.py build_ext --inplace</br>
To run: python3 run_qs.py -base 2000 -keysize 150 -debug 0 -lin_size 1 -quad_size -1 </br> (note the actual factor base per quadratic coefficient will be about +/- half of the -base parameter)

note: Only one worker at a time works for now... I need to rework that part.

To use the old PoC (that one will easily factor above 200 bit using pypy3):

pypy3 QSv3_050.py -base 6000 -keysize 200

I've added chunking. the total lin_size is calculated like this : lin_Size = lin_size * chunk_size. 
You need to keep the chunk_size as big as possible for as long as RAM allows for it and keep the lin_size as small as possible. That will yield the best performance.

Currently only one worker is supported for the main loop. You probably just want to work with multiple quadratic coefficients only once I implement parallelism. The only reason you would switch quadratic coefficient right now is if you run out of good moduli, which is unlikely to happen fast. But it's great for parallelism. 

Tomorrow I will reduce the code in the main loop as much as possible and do further optimizing. 150 bit with one worker shouldn't take more then a second. Only after that will I implement parallelism. Lets see if we can achieve this tomorrow... but I think I can.

precompute and move things out of the main loop (in addition to static typing).. just got to hammer down on that now. I know this will work. I've bled for this for over 2 years. Moment of truth now. Nothing left to lose.

Oh btw, I also noticed that small sieve intervals perform better. Its bc we've precomputed everything. Regular SIQS doesnt have this luxury. I know this will work. I will make the future gay by breaking all the PKI schemes. Thats all there is left to do anymore. I just cant get over the trauma of what happened. Everything is too late now. Too late now to change course. I will fix this world by breaking PKI schemes. 

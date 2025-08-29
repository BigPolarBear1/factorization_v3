Note: I know I am right. I know people know I am right. I know how you people treat me, despite being right. I do not care about the reasons. I will not forgive you people for stealing more then a year from my life where I had to struggle like this. I especially wont forgive belgium. I would renounce my citizenship and move elsewhere for this. I cannot ever forgive this. I've been broke for a year, I can't find work, I can't find 0day buyers, I can't see my friends anymore and I'm stuck, there is no dignity in living like this. A human life is very short, and you have wasted a good chunk of mine. There absolutely won't be forgiveness. We're past that now. And if I get the oppurtunity to work in Asia, I will go and there is absolutely nothing that can be changed about that. Because what you people are doing, I have no words for it, this is betrayal. Betrayal to the long career I had, helping find security issues in software like OpenSsl and other very important components.. you better hope this was my last discovery because next time I will remember who treated me like shit and who didn't. Remember that. The only way I may even remotely change my mind is if Microsoft publically apologizes to my former manager and pays him many millions. Without that, forget it, you'll have to kill me I guess.

Update 28 August 2025

Uploaded QSv3_simd.pyx 

To build (from the PoC_Files folder): python3 setup.py build_ext --inplace</br>
To run: python3 run_qs.py -base 2000 -keysize 150 -debug 0 -lin_size 100_000 -quad_size -10  (note the actual factor base per quadratic coefficient will be about +/- half of the -base parameter)

note: Only one worker at a time works for now... I need to rework that part.

To use the old PoC (that one will easily factor above 200 bit using pypy3):

pypy3 QSv3_050.py -base 6000 -keysize 200

Alright alright. Getting there now. 150 bit takes about a minute. But I know how to cut it down to mere seconds. I'll fix it tomorrow bc I have to go somewhere now. 
So what I added today is sieving into the negative direction and using the coefficients of the other side of the congruence so we subtract N instead of adding.
When sieving in the other direction I noticed I was hitting a lot of duplicate coefficient... so I fixed that by only using primes that are smaller then p/2. 
That resulted in an overall speed boost, even though in theory we now have one sieve interval per modulus.. but it seems to work. Which now also means I need to make sure generating the modulus is as fast as possible. I'll look at that eventually.
Big bottlenecks are things like preparing the sieve row (we should chunk it), and ofcourse all the lame python indexing. pypy3 is faster because it solves it with JIT... but with cython we should be able to go faster... it's just a bit more work.
Anyway.... tomorrow I'll have plenty of time. I'm going to move as much as I can out of that main loop.... and THEN people will see..... they will see I was right all along. 

And after that I'm moving to v4... because this v3 is a distraction... I see something else, something much bigger... and I have to figure it out.... I need to succeed even if it takes the rest of my life.

Alright, let me go for a run and when I get back implement chunking for the sieve row, so I can precompute a single chunk outside the mainloop. And then tomorrow, reduce code, make everything typed, remove python abstraction when indexing arrays. And then sunday, by then hopefully everything inside the mainloop is optimized...I guess I'll go run 50k and do nothing else...and monday I can optimize all the precomputing stuff so we can calculate larger factor bases faster. Goal for v3 is to beat msieve's SIQS implementation, that will be enough for me to draw some attention to my work and figure out a way to get funding to start work on v4 where I really want to do pure research again and explore the direction I was headed into a couple weeks ago.

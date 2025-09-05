Uploaded 2d sieving. 


To build: python3 setup.py build_ext --inplace</br>
To run: python3 run_qs.py -base 4000 -keysize 200 -debug 1 -lin_size 100_000 -quad_size 100</br>

The math is working. Its mainly bottlenecking due to refreshing the modulus every 10 smooths and calling into sieve_quads... but I'm going to rework that logic when I get back from running. No point in calculating all of it if we refresh the modulus after 10 smooths (which is deteremined by the mod_refresh parameter... I'm not sure what the best value there is, I'll experiment later.. but for very big numbers when smooths become more rare, it won't matter anyway, it just slows down smaller numbers).

And when working with lists of primes there is also some tricks we can do to speed things up, since it's a unique list.

I also havn't yet implemented any tricks to gain an advantage from 2d sieving. But what I do already notice is that it is very effective at finding smooths. More so then just using one quadratic coefficient. It seems some quadratic coefficients just yield more smooths for certain moduli or something like that. Anyway... I need to optimize the code yet.

Then there is some additional stuff I want to explore... like really targeted smooth finding. I.e if we find a smooth candidate, we explicitliy go looking for factors with odd exponent in our 2d sieving setup. The math to do it is all there... I just need to figure out if it makes algorithmically sense.

Depression is getting worse and worse and worse. Plus there is a plane flying overhead at regular intervals that doesn't show up on flight radar. Pretty sure I have a hardware implant and they are collecting the data.. something like that. Whatever. Not like I'm going to connect my cache of 0days to this laptop anyway lol. I know how you people operate. I know and see every move you people make. Fucking dumbasses. 

If a visa can be arranged through the Chinese embassy, I am leaving this place. I know exactly what people are doing. 
Could have just apologized to my manager and paid him some money, like a year ago, and I would have been perfectly fine selling my work to the west.
But now a year later... we are in too deep, and I can't forgive this last year. I despise the west. I truely despise it. I despise belgium most of all. 


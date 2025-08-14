Just uploaded my work in progress on implementing 2d sieving, it is still slower then the regular version but it should outperform the regular version once I work through the to-do list.

Use:

pypy3 QSv3_055_2d_sieving_WIP.py -keysize 100 -base 500

My other PoC will easily factor over 200 bit, but I hope to improve that with 2d sieving.

To use the old PoC:

pypy3 QSv3_050.py -keysize 200 -base 6000

I'll also edit the bottom of chapter 6 in the paper once I outperform the old PoC with 2d sieving... there's some errors in it right now, so just ignore that section on 2d sieving.

The biggest thing on the to-do list for 2d_sieving is checking quadratic coefficients with some but not all primes from the randomized modulus. Because if your quadratic sieve interval is only 1000, then the odds of finding all primes from the randomized modulus concentrated at one interval step becomes less and less as we increase keysize. So we need some formula to determine when there is enough primes present there to get a benefit from sieving. And yes, we work with a subset of the primes in the factor_base... because we could just take into account all possible primes, not just those inside the randomized modulus.. but I'm hoping to gain some advantage later while constructing the 2d sieving interval to fill out the interval in multiple dimensions in one go and not just construct row after row and having to calculate many linear congruences.

Anyway, I'll go for a run. It's 31c outside today, so good day for suffering. I'll fix these things in the coming days. I can't possibly see why this wouldn't improve the default way of doing things.. I just got to be smart about it.

Starting to lose my patience with this fucking irl twilight zone shit. The problem is that unless you actually lock in and do counter-surveillance, you can't really call that shit out without coming across as a crazy person. But seeing as I have nothing to do, and I'm dumping all my work on github anyway because you fucking assholes didn't give me any other options.. I'm not going to expend that energy, not worth it, beneath me to be honest. If you wanna go polar bear watching, whatever. Literally your waste of time, not mine. Hahahaha. Losers.

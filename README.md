Just uploaded my work in progress on implementing 2d sieving, it is still slower then the regular version but it should outperform the regular version once I work through the to-do list.

Use:

pypy3 QSv3_055_2d_sieving_WIP.py -keysize 100 -base 500

My other PoC will easily factor over 200 bit, but I hope to improve that with 2d sieving.

To use the old PoC:

pypy3 QSv3_050.py -keysize 200 -base 6000

I'll also edit the bottom of chapter 6 in the paper once I outperform the old PoC with 2d sieving... there's some errors in it right now, so just ignore that section on 2d sieving.

The biggest thing on the to-do list for 2d_sieving is checking quadratic coefficients with some but not all primes from the randomized modulus. Because if your quadratic sieve interval is only 1000, then the odds of finding all primes from the randomized modulus concentrated at one interval step becomes less and less as we increase keysize. So we need some formula to determine when there is enough primes present there to get a benefit from sieving. And yes, we work with a subset of the primes in the factor_base... because we could just take into account all possible primes, not just those inside the randomized modulus.. but I'm hoping to gain some advantage later while constructing the 2d sieving interval to fill out the interval in multiple dimensions in one go and not just construct row after row and having to calculate many linear congruences.

Just uploaded my work in progress on implementing 2d sieving, it is still slower then the regular version but it should outperform the regular version once I work through the to-do list.

Use:

pypy3 QSv3_055_2d_sieving_WIP.py -keysize 100 -base 500

My other PoC will easily factor over 200 bit, but I hope to improve that with 2d sieving.

To use the old PoC:

pypy3 QSv3_050.py -keysize 200 -base 6000

I'll also edit the bottom of chapter 6 in the paper once I outperform the old PoC with 2d sieving... there's some errors in it right now, so just ignore that section on 2d sieving.

The biggest thing on the to-do list for 2d_sieving is checking quadratic coefficients with some but not all primes from the randomized modulus. Because if your quadratic sieve interval is only 1000, then the odds of finding all primes from the randomized modulus concentrated at one interval step becomes less and less as we increase keysize. So we need some formula to determine when there is enough primes present there to get a benefit from sieving. And yes, we work with a subset of the primes in the factor_base... because we could just take into account all possible primes, not just those inside the randomized modulus.. but I'm hoping to gain some advantage later while constructing the 2d sieving interval to fill out the interval in multiple dimensions in one go and not just construct row after row and having to calculate many linear congruences.

Anyway, I'll go for a run. It's 31c outside today, so good day for suffering. I'll fix these things in the coming days. I can't possibly see why this wouldn't improve the default way of doing things.. I just got to be smart about it.

I am really depressed. Lately it is a rollercoaster... depression and anger. That is all I feel anymore. And I don't even know anymore who I'm really angry at. Guess I'm just really dissappointed how things turned out and sad due to never seeing my friends anymore. I guess when I finish this 2d sieving stuff, I'll start making some noise about it before someone steals it or something. I do not trust humans. Some humans will do anything to get ahead in this rat race. I guess sunday I'll go run that marathon, going to be a warm fucking day too, I might just drop dead from heat exhaustion. Now that would be an heroic way to go... almost as good as become part of the frozen landscape in the arctic on a mad expedition.... which I'm too broke for atm... so a marathon will have to do. I've found myself increasing my running mileage like a crazy person these last few weeks... I just mentally cannot cope anymore and I don't know what else to do to stay sane.

I GOT IT!

Use: QSv3_009.py -keysize 50

Added v3_009 ... just fixes a small bug where the modulus for y<sub>0</sub> wasn't calculated correctly... next version will include transfering the result to a primefield..

If you run it, 50-bit is still near instant. But at 60-bit the cartesian product gets out of control, even with prime lifting.
However, in number field sieve, they transfer the result to a primefield, which has only two roots. So that's what I will do.
My hope is that that will yield a correct root each time. 
If not, I will need to dig in some more and have a look at the math.

At the very least this confirms that just iterating i-values is a valid strategy, as long as both y<sub>1</sub> and y<sub>0</sub> come from a sufficiently large modulus. Then it has garantueed roots amongst its members that yields the factorization of N.

In a sense, this whole setup is very similar to number field sieve, but just different enough that I struggled for some months to make the right connections. In numberfield sieve we have just one polynomial congruent to N. Here we basically have fragments of quadratics mod p<sub>i</sub> that we then reassemble. I see how it relates. But it's pretty complicated. Whoever this Carl Pomerance was, that was some really intelligent guy to come up with these things (Number field sieve and Quadratic Sieve).

I'll try to implement that tomorrow. Tired today from all the stress. 

I really hope moving those results to a prime field and then taking the root with tonelli fixes this last issue... then I am finally done. If not, I'll keep going... but I really need a breakthrough right now. Life has not been easy at all. Just haunted every day by memories of better times, people and places I can't go back to anymore. It truely is the worst type of torment.

I guess to move a result to a primefield, you would choose a handful of large primes close to eachother. Then calculate the difference between those prime moduli and the composite one... and find some multiplier on that difference so that the legendre symbols of the result are squares in those primes... that should then be a good indication it is correctly adjusted. Something like that. Yea yea, should be easy enough. I kind of hate that I am not well versed in the formal language of all this math.. but I understand the numbers.. people will see soon I guess... my amateuristic paper, may get easily written off due to the lack of formal language... but code doesn't lie. And I'll prove it in the coming days. It doesn't matter that it is still slow right now, because I know I am on the right path and nearly there now... all part of the grind.

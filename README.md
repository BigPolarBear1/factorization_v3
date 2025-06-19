I GOT IT!

Use: QSv3_008.py -keysize 50

Added v3_008. Just quickly fixed the debugging output to show the actual relative i-value, because it was showing y<sub>0</sub> in 007 by accident.
Also prints the size of the Cartesian product now and enabled lifting of prime moduli.

If you run it, 50-bit is still near instant. But at 60-bit the cartesian product gets out of control, even with prime lifting.
However, in number field sieve, they transfer the result to a primefield, which has only two roots. So that's what I will do.
My hope is that that will yield a correct root each time. 
If not, I will need to dig in some more and have a look at the math.

At the very least this confirms that just iterating i-values is a valid strategy, as long as both y<sub>1</sub> and y<sub>0</sub> come from a sufficiently large modulus. Then it has garantueed roots amongst its members that yields the factorization of N.

In a sense, this whole setup is very similar to number field sieve, but just different enough that I struggled for some months to make the right connections. In numberfield sieve we have just one polynomial congruent to N. Here we basically have fragments of quadratics mod p<sub>i</sub> that we then reassemble. I see how it relates. But it's pretty complicated. Whoever this Carl Pomerance was, that was some really intelligent guy to come up with these things (Number field sieve and Quadratic Sieve).

I'll try to implement that tomorrow. Tired today from all the stress. 

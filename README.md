# factorization_v3

Final version, will be released shortly.
Improves on factorization_v2 due to the realization that if we know how many times N we are subtracting from each quadratic coefficient mod p, then we can eliminate a while bunch of primes that the other side of the congruence can possibly be divisible by, since both sides need to be quadratic residues... see quadratic reciprocity. And hence in theory we should be able to build up congruences by grouping primes together one both sides using quadratic reciprocity, but I'm still working out the details.. however, I already know it can be done... just struggling with stress and depression I guess.

So in v2 I showed how to group coefficients mod p together such that subtracting x * N results in a number divisible by the coefficient's primes.
For example if we look for coefficients with 4 * N subtracted using our example with P = 41 and Q = 107 (PQ=4387) we get:

x^2 = y^2 + 4387 * 4

or (37 * 4)^2 = (2 * 3 * 11)^2 + 4387 * 4

Lets say we only know one side:

x^2 = (2 * 3 * 11)^2 + 4387*4

Now this equation, whatever modulus we reduce the right side to, it HAS to be a quadratic residue, else it will break the left side.

And ofcourse visa-versa.

So this reduces it to finding combinations on both sides that are quadratic residues in all mod p. 
And since this can be represented by legendre symbols... it can be reduced to a matrix in gf(2) ....

I cant see in my head why I cant solve it this way. I'm just so fucking depressed. These last few days, I literally feel like wanting to kill myself.
Guess being broke and hopeless about the future does that to someone.... it's just hard man... just so depressed it physically hurts.

People better hope all this traffic I have been seeing ever since the 23rd of August is just stalkers and stupid infosec idiots who google me to feel better about their own shit life (yea, my entire career people wanted nothing more then for me to fail, I know how it is).
That that is all there is too it... because I swear, if I am right about my math and people knew about it for months but treated me like "the enemy" ... I am going to be in the worst mood ever. I will literally go live in a hut in the woods and never talk to a human being again. Anyway... quadratic reciprocity... lets finish this this weekend... I have a feeling I am about to "find out". It would be nice if we could use legendre symbols instead of exponent matrices... for so many reasons. I just need to work out the exact details and write the code. But in my head I cant immediatly spot why it wouldn't work.

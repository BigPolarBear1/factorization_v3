Just uploaded v3_004:

useage: python QSv3_004.py -keysize 40

Just uploading my work in progress. This one will easily factor 40 bit. Which is way too slow. But I have finally gained a good understanding.

So we need to maximize the modulus of two selections at different iN values, while minimizing the iN value (and the bigger that difference between i value and modulus, the more likely we have a square relation).
I have an idea to achieve this using linear algebra over gf(2). I think this representation is perfect for it.
So next version will include some linear algebra....

Factorization hasn't fallen yet today, but I know what I've got, and in the coming days, it will fall. So go to hell. Wait until you see my next version with linear algebra lol. 

Update: Hmm. Thinking about it some deeper. Perhaps linear algebra would be overcomplicating it. We can simply represent our iN datastructure in a way we can quickly query for small iN values... by extending each of the primes up to a certain predefined bound.
Fuck, I am so tired and sleep deprived today. I should go for a run soon, get some proper sleep and try again tomorrow. I KNOW I GOT IT NOW. I just need to fix this fucking code.

Fuck, I just need to change that iN datastructure to some square-ish looking datastrucutre by extending all the primes up to a certain bound (or sieve interval). Where the bound is the max iN value we are interested in. Then that solves finding small iN values and reduces the problem to simply finding a small coefficient and then finding another selection at a different iN value while maximizing the modulus. Something like that..... fuck you insomnia.

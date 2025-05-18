# factorization_v3

Eureka. I think I have finally managed to connect all the dots now. 
The whole problem is just this: x<sup>2</sup> = a mod N, now find a pair of roots that isn't of the form x and N-x. Any other pair of roots will yield the non-trivial factorization of N.
We can't do calculations mod N, since it requires knowledge of the factors of N. However, we can do analogue calculations mod p<sub>i</sub> where p is prime.
And I believe I just found the last piece of the puzzle to get that approach to work, building on two years of research.

Going from v2, to v3 has been the most difficult and chaotic so far... but with some luck, v3 will come online soon now... I want v3 to break factorization, to finish this chapter in my life and move on to the next big challenge. I'm not settling for less. I don't care about improving quadratic sieve... maybe 30 years ago people would have cared about that.. I have to break it completely, it is the only way any of this will have mattered. 

Just uploaded v3:

useage: python QSv3.py -keysize 16

Only use a keysize of 10-20.

Its super slow.

This is as intended! Next I need to think about a fast way to retrieve small coefficients from the xN hashmap (its not really a hashmap atm, but I should probably turn it into one to solve this issue or do something similar the way we find the first small coefficient).
And right now it only continues if every coefficient mod p has a valid xN counterpart, but it doesn't have to, we can simply adjust the modulus sometimes.

Also need to add worker support and coefficient lifting... but hey... the math checks out, and I know what this means, so go to hell because tomorrow it is finding out time.

The optimized version will come tomorrow. 
This is simply a proof of idea. 

I'll spent the rest of the day working on my paper, which I hope to release also this weekend or next week. I'll begin optimizing this uploaded PoC tomorrow. The performance ceiling will be incredibly good, it will easily surpass factorization_v2 and then some. And we completely eliminate the need for smooth finding. Which as far as I know, is novel and a breakthrough. So yea, go to hell. Could have just treated me with the tiniest bit of respect... but noo.. you had to do it like this.. so be it, pay the price then.

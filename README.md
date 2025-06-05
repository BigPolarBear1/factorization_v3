Just uploaded v3:

useage: python QSv3.py -keysize 16

Only use a keysize of 10-20.

Its super slow.

This is as intended! Next I need to think about a fast way to retrieve small coefficients from the xN hashmap (its not really a hashmap atm, but I should probably turn it into one to solve this issue or do something similar the way we find the first small coefficient).
And right now it only continues if every coefficient mod p has a valid xN counterpart, but it doesn't have to, we can simply adjust the modulus sometimes.

The optimized version will come tomorrow. 
This is simply a proof of idea. 

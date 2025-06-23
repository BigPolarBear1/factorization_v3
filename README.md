Just uploaded QSv3_010.py

Useage: python3 QSv3_010.py -keysize 40

It is very slow still. It is as intended. Just uploading my work in progress for the day.
However this demonstrates indexing a hashmap by relative i-value. So that any coefficient pairings we find are garantueed to be at the correct i-value.


There is two loops:

1. The outer loop collects coefficient pairings for a relative i-value. (this does not matter, since we WANT those i-values to be small anyway)
2. The inner loop then finds small coefficients in that collection.

Step 2 needs to be approached completely differently. As depending on the size of the modulus, there will be many good coefficients that garantuee a result. 
We shouldnt iterate there, that's horribly slow. I have an idea using the math I figured out earlier in the weekend (I realized the relative-i value is just the quadratic term's coefficients from both sides of the congruence multiplied together).
So that will be reworked in the coming days.

Secondly, we should take the square root over a prime field. This reduces the size needed for the modulus and will also massively boost performance. 

Give me a few days now to address these issues, and you'll see ;).

Update: Ah lol, I see how it works now. We just need to get the modulus big enough, then most of those coefficient combinations will work since we are now garantueed to be at the correct i value.

I'll fix it tomorrow. Enough work for today. I know I have it now. Few more day doesn't matter. 

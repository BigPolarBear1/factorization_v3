Just uploaded QSv3_010.py

Useage: python3 QSv3_010.py -keysize 40

It is very slow still. It is as intended. Just uploading my work in progress for the day.
However this demonstrates indexing a hashmap by relative i-value. So that any coefficient pairings we find are garantueed to be at the correct i-value.


There is two loops:

1. The outer loop collects coefficient pairings for a relative i-value. (this does not matter, since we WANT those i-values to be small anyway)
2. The inner loop then finds small coefficients in that collection.

Step 2 needs to be approached completely differently. As depending on the size of the modulus, there will be many good coefficients that garantuee a result. 
We shouldnt iterate there, that's horribly slow. I have an idea using the math I figured out earlier in the weekend (I realized the relative-i value is just the quadratic term's coefficients for both roots (aka factors of N) multiplied together).
So that will be reworked in the coming days.

Secondly, we should take the square root over a prime field. This reduces the size needed for the modulus and will also massively boost performance. 

Give me a few days now to address these issues, and you'll see ;).

Update: Ah lol, I see how it works now. We just need to get the modulus big enough, then most of those coefficient combinations will work since we are now garantueed to be at the correct i value.

I'll fix it tomorrow. Enough work for today. I know I have it now. Few more day doesn't matter. 

Update: You know, I understand it now. In theory, if we take square roots over a prime field, and then keep lifting... eventually we will hit a square in the integers. I'm just curious, why does number field sieve, sieve, when you can just set it up like this? I'll get that square root taking code to work... I guess potentially some of those coefficient combinations, they may grow to be incredibly big numbers... I've kind of figured out already, if the coefficient and i-value is kept small, we are garantueed to have a square relation with small-ish numbers... so there is a way to mitigate that.. but I'm just curious how big some of these will grow until they hit their square relation. 

Update2: Yes yes, this works now. Also funny how calculating the square of the other coefficient is the discriminant formula. So anyway, since we know this i-value represents the quadratic term's coefficient, we can now define a polynomial ring to take a square root over. Hence this solves it. This completes my work finally. I think for the rest of the day, I'll edit the paper to atleast include these findings, so the paper is atleast factually correct and somewhat complete. I have something to do this evening so not a lot of time today. Then tomorrow I will implement the code to take the square root over a polynomial ring.

Update3: I modified the paper somewhat. It is super rushed. Had little time today sadly. I've also figured out the math for taking square roots over polynomial rings in a finite field.... so I'm hoping that will work now. The thing I didn't understand until these last few days, is how polynomial rings work. And that was also a thing in number field sieve that I struggled with understanding the most. But now I get it. I hope it is no smooth sailing the coming few days to finish it. I feel like i've grown older, a centurie, just working on this problem for 2 years, starting from 0 math knowledge. I know when I finish this, I'll probably take a break for a few months, just spend some time learning math from books, not having this pressure to solve a problem and get my life back together... and then hopefully, eventually I will find the energy to either continue my work on factorization (if I feel like more progress can be made) or find a new hard problem to work on and apply everything I've learned so far (discrete log problem sounds interesting... same modular type of problem that I so enjoy working on). I've been unemployed for 1.5 years now. Now income. Living with my parents. I am mentally truely at the end of my rope. It's the hardest thing I have done in my life. The math isn't hard... it's just a process... that I know from bug hunting... but it's everything else... my life's circumstances... everything that happened, the trauma.... just everything. I want to fix this world with math somehow, perhaps if I throw my life at this relentlessly, eventually I'll succeed. Nobody should have to suffer needlessly, let alone die at the hands of violence and conflict... which these days seem to be spiraling out of control. This world is broken, and it needs to be fixed.

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

Update: Just messing around some more. I think the fact that both coefficients must share a common root, can be used to adjust the quadratic term's coefficient. I could alternatively just go full on number field sieve and work modulo a polynomial ring. But I'm not entirely convinced we *have* to take that route. Plus we are already working in a polynomial ring... since all the results will be N mod m if we complete the quadratic. Anyway, we know we got the correct quadratic coefficient once both coefficients share the same root in the integers... or any mod p.... so that, I am fairly confident I can leverage to finish my algorithm. Lets see tomorrow. I really regret not having a proper math education... I feel like I'm struggling with stuff that's really basic and would be immediatly obvious to someone more proficient with math.

Hmm... slowly starting to understand it now. Understanding how those quadratic term's coefficients work really was key. The only thing I struggle with is, in number field sieve, the polynomial ring is made from irreducible polynomials... and I don't want to change the shape of my quadratics because it shifts everything, and its honestly annoying. And for any given coefficient mod m we find, any of the roots will produce N mod m anyway.... so we do have polynomials congruent to N (just in mod m instead of in the integers). However, there probably is some calculations using the derivative to find a good correct quadratic coefficient. In our current setup we are garantueed the correct quadratic coefficient mod m... but after that we need to adjust it to the integers, so that the discriminant formula produces a square in the integers, not just mod m. Isn't that what number field sieve also does by taking a square root of a polynomial ring using derivatives... atleast that's how I'm currently abstracting it in my head? I'll bash my head against this tomorrow. Close now. Very close. Shouldn't be too long. 

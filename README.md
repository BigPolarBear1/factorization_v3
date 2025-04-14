# factorization_v3

Dropping this week.

Got the matrix math working now.

So one way of doing it... just factor what you can over the factor base. Then the remainder can be add to the matrix row as a jacobi symbol modulo p1 * p2 * p3, ... ,.. 
So that way we don't need to completely factor it over the factor base.
Then the hope is that the part that isn't factored over the factor base has similar factors when dealing with larger numbers. Or at the very least it allows us to find squares in our finite field... which can be useful bc then we can just take the square root in said finite field.. and hopefully that will then yield the correct square root. It's the best I can do at the moment I think. I wish I had access to a professional number theorist to bounce some ideas off.
I studied number field sieve again all day yesterday, and aside from doing the above, it's hard to do the exact same as number field sieve bc in our example we don't have polynomials with roots and coefficients congruent to N... we have tiny pieces of it in a way.. so it's completely different. And hence many of the tricks are kind of difficult to adjust to my own findings. I would need to do some further studying.

So anyway that works in my code. It will find the square relation while still being congruent mod N on both sides using that jacobi symbol.
I'll plug that into my v2 logic next.

Then after that I can see if it can be further improved by just taking the square root over a finite field incase the total product isn't a square in the integers.
The trick will be to have the result still congruent mod N on both sides... but I can figure that out, I'm sure.

I'll do that tomorrow. 
My head is just not in a great place. These financial issues are really just starting to give me a headache. 



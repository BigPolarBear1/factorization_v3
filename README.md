# factorization_v3

Should be any day now.... trying to add to v2 in a meaningful way has been a bit chaotic. Too many different ideas. 
I do honestly believe I got it now. I hope to get it done this week. Next week I'll start doing other work aswell, because I am completely broke and I need an income.

Speaking of an income... I am also willing to take a job, I have a decade of experience in vuln research and found many bugs in windows 10, secure channel, kerberos, ipsec, openssl, etc.
Will relocate anywhere. Am a Belgian citizen with no criminal convictions. but if a pathway is open I will become a citizen somewhere else... as Belgium is a dead end for me, especially having no formal education and very niche work experience.

Email: Big_polar_bear1@proton.me

I can also do contract work... and if needed also relocate for the during of the contract. I can move and start immediatly. Quite desperate to restart my career at this moment to be honest.

Bleh.

Now I seem to have got it.

So we can reduce things to a finite field of small primes.

I.e p=107 and q=41 thus N=4387

The correct coefficient is 148 and 66 on the other side of the congruence. 148^2 = 66^2 + 4387*4

Now, we lets say we find coefficient combinations in prime pairs for example.

Lets take 11 and 13.

148 % (11*13) = 5

So lets say we found 5 in the finite field of 11 * 13.
Now we can find smooths in a finite field. Subtract 4 * N = 5^2 - 4387 * 4 = -17523 % (13 * 11) = 66.
66 factors over 2,3,11. It is not a square, but we can use it as a smooth relation. If one of the primes in the fininite field, divides the integer when x * N is subtracted, then reducing it to the finite field, will keep that divisor. And often times, it also includes the other correct divisors, as is the case here.

I wonder... I should then be able to extend the finite field with primes of the divisor... as long as the divisors are in the factor base.. I should be able to do that... until the coefficient grows large enough, at which point we have constructed a smooth. Hmm... I like this direction of thinking... lets get the math and code done this weekend... I am mentally tired.

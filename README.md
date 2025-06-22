Announcement: As of today, the 22th of June. I made the final breakthrough that was required and constructed proof of concept code.

The thing that I missed before was how the coefficient for the quadratic term relates to how many times N divides the difference between both squared coefficients of the linear term.
With that, knowledge, I finally succeeded in creating a hashmap that can now be queried by using coefficients for the quadratic term as hashmap index.
This then returns possible coefficient pairings. No longer do we now need to bruteforce this i-value that I talked about in the paper (which is just the coefficients for the quadratic term on both sides of the congruence multiplied together) ..
now we find pairs of coefficients at the garantueed correct i-value (because knowing it represents quadratic term coefficients it was easy to figure out the math for this).

Hence this solves it. This solves the very last issue I had. 
I need one week max now to finish the PoC and I'll have to massively refactor the paper too. 

Expect the final v3 version to be uploaded within the timeframe of a week.

Thanks for your patience, and if I die before publication, I garantuee you it was the americans.

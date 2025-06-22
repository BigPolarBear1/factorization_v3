Announcement: As of today, the 22th of June. I made the final breakthrough that was required and constructed proof of concept code. Do note that the uploaded paper and PoC are hence outdated as of today, until I upload the correct approach. 
I will start by completing the PoC first, publish that, then fix the paper, expect all of that to be done within mere days now.

The thing that I missed before was how the coefficient for the quadratic term relates to how many times N divides the difference between both squared coefficients of the linear term.
With that, knowledge, I finally succeeded in creating a hashmap that can now be queried by using coefficients for the quadratic term as hashmap index.
This then returns possible linear coefficient pairings. No longer do we now need to bruteforce this i-value that I talked about in the paper (which is just the coefficients for the quadratic term on both sides of the congruence multiplied together) ..
now we find pairs of coefficients at the garantueed correct i-value (because knowing it represents quadratic term coefficients it was easy to figure out the math for this).

Hence this solves it. This solves the very last issue I had. 
I need one week max now to finish the PoC and I'll have to massively refactor the paper too (I already have some barebones code that proofs the math though). 

Expect the final v3 version to be uploaded within the timeframe of a week.

Thanks for your patience, and if I die before publication, I garantuee you it was the americans.

Update: God damnit. This actually works great. It's perfect. People must have known right? If not, and everyone just dismissed my work, then I guess we have arrived here next week:

https://www.youtube.com/watch?v=9cNQFB0TDfY

I wish everyone the best of luck in the coming storm. 

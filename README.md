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

Update: God damnit, I am seriously nervous. Something feels off. Have I truely snuck  under the radar and made a major breakthrough? Like, any math educated person, who would have looked at my work, would have realized what was going on there, right? Then why did things go down the way they did? I just have this nagging feeling that something is off. Factorization_v2 was so close.. I just didn't realize that damned coeffiicent for the quadratic term.. that was the only clue I had missed for months, but smart people must have looked at factorization_v2 and immediatly made that connection, right? So what the fuck is going on? Perhaps this was supposed to go down this way. People gave me no other choice. No way out. So what will happen, will happen now. And also, the future will be gay as hell and you haters cant stop it, hahaahaa.

Update: Doing a bit of work today. Its coming along really well. Also figured out, the best way to construct the hashmap is to index it by both coefficients of the quadratic terms from both sides of the congruence multipled together. Aka, what I would refer to as the relative i-value in the paper. So our main loop, we just pull coefficients pairing based on that relative i-value.. and we're only interested in small i-values anyway (since the bigger that value, the bigger the modulus we need to work with)... so it doesn't matter much that we're looping here. Then the ONLY thing that remains once we got coefficients pairings, is finding a small enough coefficient on one side of the congruence... which then also automatically means its paired coefficient is going to be relatively small, since we are working with small relative i-values ... which also means its going to fit inside the modulus. That should be pretty fast. Then ofcourse one can come up with all kinds of ways to optimize the hell out of this. Anyway, taking it easy today, feeling a bit depressed. Will probably upload a PoC tomorrow, don't want to pressure myself into rushing out a PoC today... been going at this for over 2 years now... a day more or less won't matter.

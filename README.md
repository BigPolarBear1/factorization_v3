Just uploaded v3_004:

useage: python QSv3_004.py -keysize 40

Just uploading my work in progress. This one will easily factor 40 bit. Which is way too slow. But I have finally gained a good understanding.

We need to find a small coefficient while maximizing the modulus and then try and find a small iN value where each solution maps to another solution.
The above code does this, but pretty poorly.
Next this will need to be replaced with linear algebra. 
My next version will do this. Pretty sure I know how to do it now.

OH!! I just went for a run, and I suddenly had the most brilliant idea! let me try to write it before I pass out, otherwise tomorrow. Waging cryptologic warfare against my enemies (the fbi and Belgian justice department) sure is exhausting. But I will triumph in the end, because no human can compete against a polar bear, hahahha.

Shit's getting too stressful. When I'm doing math, I do math 100%, I have been doing math at 100% for 2 years now. And I put massive amounts of pressure on myself to succeed at it. And honestly, right now, I can't deal with the legal harassment from Belgium and those fbi losers. I just need to get out of this place, there's plenty of countries where the US has no influence at all for me to go to.


TO DO: So the math in the paper and PoC, it's all correct, it works. We can find coefficient pairs that are congruent mod N in GF(m) (when squared). And we also know the conditions it must meet to be also congruent mod N in the integers. Figuring out that part was the hardest part. That is now entirely finished. Now part 2 of the challenge begins. The question now is, how can we use these findings to construct an efficient algorithm? I'm fairly sure there is a similar method to using linear algebra as Quadratic sieve does, to combine coefficient pairs to generate a congruence mod N in the integers. I just got to find my focus again for a few days and its done. Just all the stress of the legal harassment, and unemployment for 2 years, not being able to see the friends I had known for 4 years anymore...  I feel like i am fighting an uphill battle against the entire world. Almost there now... I just now it, and people will see soon. If we can work with just coefficients, then it will be exponentially faster then quadratic sieve, and I know I'm right, I just 100% know it.

Aha, so what's really at the core here, we need a quick way to figure out what the iN value should be for a given coefficient. If we can figure that out quickly, everything else will follow. Even if a coefficient pairing is only congruent mod N in gf(m) ... we can still find it by looking at gf(m*i) easily... AS LONG as that iN value is correctly. I know there is a way to do it. If I find this, factorization falls. Watch me. A few days at most now.

Update: I got it. I figured out how to quickly calculate the correct iN value for a given coefficient. And after that it is simply a matter of adding the modulus to a squared coefficient, until it becomes a square in the integers... or just working with big moduli.. as that basically garantuees us to find the correct results once we find the correct iN value. This can be done after all, I knew it. Prepares yourselves, haha. Factorization will die.

I'll go for a run an explain the math between those iN values in the paper when I get back. Finding the correct iN value for a given coefficient, that's really the core. Solve that and factorization falls. If I survive my run, for some reason my legs keep cramping up at night really badly (resulting in pain throughout the day) and I also keep feeling dehydrated no matter how much I drink. Must be the heat. I'm a polar bear. Normally I would go hike in the arctic around this time of year... but I'm broke.... so yea....

I dont quite understand why literally nobody seems interested in this work... for like a year now. People must see, that I'm doing all the correct things here, not going off into magical tangents. Everything I do and say adds up, all the math checks out, its all there. Sometimes this becomes mentally really challenging... like you legit start questioning your own sanity. But I know I'm there now. The biggest struggle is behind me. I've climbed over all the tall mountains, now its just a couple of rolling hills to get over the finished line. A mere days away now.

Update: Added a bunch of stuff to the final chapter. So far it shows how the correct iN value is determined for a given coefficient. I've already worked out some of the math to figure out that value, given only one coefficient ... but will add tomorrow as its past midnight and i'm tired.

Hopefully tomorrow I'm done... I don't get it... because I now, I'm right there now, about to succeed.. and its like nobody else sees this but me.. and it legit makes me question my own sanity, but I know I'm spot on, I just know it. Some days it is really difficult, just relentlessly believing in myself while nobody else does, and working on this for 2 years without income or job. There was a lot of dark days... probably some of the darkest days i've had in my life... but I know I got it now.. so lets get some sleep and break this shit tomorrow.... people, they will never believe me anyway, until I show a PoC that is really fast.. that is the only thing that will convince people. Until that happens, in their human stubborness they will refuse to believe even the remotest possibility that I might succeed... which is while this entire situation is so god damn painfully funny and sad at the same time. 

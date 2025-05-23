# factorization_v3

Eureka. I think I have finally managed to connect all the dots now. 
The whole problem is just this: x<sup>2</sup> = a mod N, now find a pair of roots that isn't of the form x and N-x. Any other pair of roots will yield the non-trivial factorization of N.
We can't do calculations mod N, since it requires knowledge of the factors of N. However, we can do analogue calculations mod p<sub>i</sub> where p is prime.
And I believe I just found the last piece of the puzzle to get that approach to work, building on two years of research.

Going from v2, to v3 has been the most difficult and chaotic so far... but with some luck, v3 will come online soon now... I want v3 to break factorization, to finish this chapter in my life and move on to the next big challenge. I'm not settling for less. I don't care about improving quadratic sieve... maybe 30 years ago people would have cared about that.. I have to break it completely, it is the only way any of this will have mattered. 

Update: Making some good progress now. I think I have finally smashed through the final wall now. Just need a few good days without insomnia or depression.
Definitely 100% can make big improvements to v2 already. But I'm hoping this will break factorization now... we'll see.
I don't quite understand why nobody seems to genuinly care, considering what is at stake.. but it is what it is. Maybe if they hadn't fired my manager, I would have given second thought, but now I only feel pain and regret. 

Update: I dont know, I think I can finish it now, but my productivity has crashed. Just got this awful feeling. Something doesn't seem right. But I know these feelings and suspicions are likely just the result of struggling with depression and anxiety. One can easily begin to drift away from reality dealing with hopelessness for a very long time. I mean, it doesn't really matter I guess.. I know I will solve this eventually, either that or someone else will beat me too it, that is going to be the only outcome. Factorization must fall. I refuse to live in a shit world without factorization. Everything just going to shit. I'm going to fix everything.

Update: I need to finish this asap. Just starting to get a bad feeling. Anyway, its fairly easy to improve v2. We don't need square finding. Given a root and coefficient combination, you can take the derivative to find the coefficient for the otherside of the congruence, and then just go straight to taking the GCD of both coefficients and N. I'm still trying to figure out the damned implementation details. Any day now.. just having this really awful fucking feeling and I can't focus anymore.

Some type of betrayal in the air... I have a sixth sense for these types of things. People like me, we're always going to be set-up to fail, no matter what we do, especially in the current political climate. It doesn't matter. Factorization will fall. It is more important that it falls then one person's life or happiness. For some reason, defeating factorization is the most important thing in the world. I don't understand it yet. Or perhaps I am simply trying to find greater purpose in life in reaction to traumatic events. Ah, whatever. Lets see if I can finish the paper already this weekend, upload it... then I can take my time with the PoC, get it perfect.

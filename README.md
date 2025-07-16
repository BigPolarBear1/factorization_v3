Update 16 June:

Uploaded QSv3_036.py 

use: python3 Qsv3_036.py -keysize 120
This one massively improves the building of the iN map. Even for a factor base of 1000, it takes only a few seconds. 
Also sped up the lifting code a little by using a faster root finding method.

And I made some progress toward removing trial division. It will now only factor a smooth divided by the known factors if the log base 2 is an integer (power of 2). I tried implementing lifting for powers of 2 today, but in my honest opinion, 2 is not a real prime... and I'm not even sure if it is worth the trouble adding all these special cases just to be able to handle powers of 2. I might aswell calculate log2 to see if something factors over a factor base and then check if its log is an integer (meaning the remainder can be factored by powers of 2). I will think about this some more in the future.

Anyway, I lost a lot of time today messing with powers of two only to end up deleting all that code. But we're already close to the performance of v2. We can factor 120-bit in about 2 minutes now. 
Tomorrow I will begin refactoring ALL of that smooth finding code. The idea is really good and works really well, but my current implementation is very sloppy. I know how to fix it though and will begin working on it tomorrow. The cool thing is, that even with this code, we can now pull many smooths just one iN value (or quadratic coefficient) ... so it does work... I just need to fix that code and be smart about it.

--------------------------------------------------------------------------------------------------------------------
I know shit is about to hit the fan now, for real.
I have been working on this for 2 years. 
2 years ago, I told the msft people who were firing me that I was working on factorization (the lawyers, investigators and HR who was making the decision after my mental breakdown due getting harassed in Redmond).
Shortly after getting fired, I told the FBI many times I was working on factorization. I made it very clear to them during multiple meetings. Yet the state department failed to arrange a visa.
I had no choice return to Europe. 
Upset that I had to leave all my friends behind, that I had known for 4 years, I then for the first time, told the internet (on twitter) that I was working on factorization.
Yet, european law enforcement ended up chasing away potential buyers in europe, threatening them with sanctions for even talking to me. Which to this day, I still do not understand how you can be so dumb.
Everyone knew I was working on this. Yet no way out was given.

I simply did not have a choice. So if again, like last time, when I was forced to drop 0days due to similar circumstances, you again start demonizing me, come do it to my face... ofcourse you would piss yourselves and I would 100% punch you. Cowards.
I know how negatively this will affect my life. I just had no other choices. The only other choice I had was not working on factorization. But I fell in love with factorization, and that's the end of that.
I hope you all got what you wanted, and I hope you will all burn in hell, because I won't ever forgve this.

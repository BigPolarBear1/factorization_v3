Note: That last chapter in the paper is becoming a bit of a frankenstein chapter. I will first finish the PoC, then redo that entire chapter. Most of the math is already in there... its just not pretty looking and all over the place right now.

Just uploaded v3_005:

useage: python QSv3_005.py -keysize 30

Just uploading my work in progress. Bit slower then the previous version. We basically started over again, but this time using the correct simplified math.
We start with a small coefficient, alpha... make sure its a quadratic residue of N... if it is, that means, it must be found as a coefficient solution for every modulo prime. So for each prime, we check at which iN value this coefficient exists... and then finding the other square basically comes down to finding another iN value where everything maps to a solution. 
Next we are going to get rid of that loop, this code has only one loop (excluding the loop for alpha..that one matters less) and we're going to get rid of it... watch what happens in the coming days :).

Update: Almost there... proper PoC should drop any day now... 

I think tomorrow.... I know how to solve it now. If alpha is a small coefficient... I'm already doing it in the code but breaking after retrieving the first iN value... but it will actually have multiple ones (i.e in mod 13, the coefficient "1" will appear at multiple iN values)... and all of this is rather fortunate bc it lets us do something..... you will see. Big day tomorrow. I know I've been saying this all week now.... but that's bc I'm an inch away from breakthrough now.... just took a few days longer then I had hoped.. 

Me, becoming a threat to national security, like the polar bear I was supposed to be. Hahaha. Fuck you all.
Fuck the fbi and fuck the Belgian justice department for their legal harassment. I didnt start this war, you people did. We're deep into cryptologic warfare now, no turning back anymore. This is my kamikaze run on factorization, I will succeed and I have nothing left to lose, you people already destroyed everything.

What people may or may not understand is that finding an iN value where every solution mod p, has a mapping for every possible prime p (up to infinity in theory)... is that it gives us an instrument to sieve for the correct iN value very effectively... hence, this is why I have already won. Ofcourse, this is but a first battle won, in cryptologic warfare, there is no stopping until the complete annihalation of the enemy. Shouldn't have fired my manager. That one was your fatal mistake in a series of many mistakes.

Update: Slow sunday.... just feeling particulary depressed today. People, they really have destroyed everything that was good in my life. It is very difficult, thinking back about better days. 

Update: Wait actually, this can be easily used for smooth finding on steroids with everything I understand now... let me upload that. Fix the paper. And then... for what I *really* want to do, I will do as part of factorization_v4. Let me skip my run today and just work through the night to get this done asap.

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

I know it can be done.. building smooths. I should have gathered all the info I need to be able to do it. Now its just a matter of doing it. Lets see this weekend. You know what people dont understand about me? I will literally spent the rest of my life on this problem just to solve it. Because people don't know my level of determination. I will gladly sacrifice everything to do this. And I will succeed no matter the cost.

The very moment I solve "smooth building" factorization falls. And I will not stop until I figure out how to do it, because everything I know, everything I learned, indicates it is possible. It's that same intuition screaming at me when I look at "code" and know there will be a bug. I will succeed. I have decided to make this problem my life's work. This is the problem I fell in love with, and I will never stop for the rest of my life. And all the people who don't believe in me, it's their loss, because they could all have gained so much. It is inevitable that I will succeed. And I just know it.

You know, my whole life, I have just seen bad things happen to good people. I'm not refering to myself, I'm not a good person, I have a lot of demons haunting me. But good people, who selflessly believed in me, and supported me, not just through words but also action. And time and time again, I have seen nothing but misfortune happen to good people. And none of it is right. This entire world is upside down. And the only way, to truely change it, is by breaking factorization. To make a better world. And I am sure my stalker would call that a delusion, but such are the cries of lowly men who have no aspirations in life. I will succeed, because I believe it is the only way to make right all the wrongs. I cannot accept defeat. And in the offchance that I don't figure it out this weekend, and have to prioritize making money again for a little while, every second of the day, my mind is not occupied with making money, it will be grinding this problem. I am a polar bear, a polar bear never quits the hunt, until my last dying breath.

Update: Eureka. I knew it. I knew there was something there. That should give me a chance to construct smooths.. starting with a small finite field, and then expanding it based on the factors we see. Kind of strange how this works. Feel like there is something deeper here, more fundemental I must explore eventually.

My work is almost done now.... semiprimes all around the world will tremble in fear soon, haha. I don't quite understand fully (I have some intuition about it and a possible theory) why this reduction to small finite field yields other primes factors of the smooth candidate... but it works, over and over again at such consistency that I cannot attribute it coincidence. This is my way to crack this problem wide open, I know it. It has been a chaotic journey, but this has been the week that I finally slayed the dragon. It is just a matter of getting the code ready and figuring out the minutia of the algorithm. Prepare yourselves for factorization! It is coming and it is coming fast! mwuhahaha. 

You know, when you are deprived of a life by an industry. Unable to get anywhere in life. You just got to do whatever it takes. And if the cops come, you just fight as if your life depended on it, bc it does. And that is really all there is to it anymore. its just the way it is, and I'll never see my frieds anymore bc of microsoft.

I rather have a few more years of dignity and go out in a blaze of glory then just slowly fade away like this, denied a life of dignity, being mocked by shitheads.

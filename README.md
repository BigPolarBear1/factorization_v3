# factorization_v3

Final version, will be released shortly.
Improves on factorization_v2 due to the realization that if we know how many times N we are subtracting from each quadratic coefficient mod p, then we can eliminate a while bunch of primes that the other side of the congruence can possibly be divisible by, since both sides need to be quadratic residues... see quadratic reciprocity. And hence in theory we should be able to build up congruences by grouping primes together one both sides using quadratic reciprocity, but I'm still working out the details.. however, I already know it can be done... just struggling with stress and depression I guess.

So in v2 I showed how to group coefficients mod p together such that subtracting x * N results in a number divisible by the coefficient's primes.
For example if we look for coefficients with 4 * N subtracted using our example with P = 41 and Q = 107 (PQ=4387) we get:

x^2 = y^2 + 4387 * 4

or (37 * 4)^2 = (2 * 3 * 11)^2 + 4387 * 4

Lets say we only know one side:

x^2 = (2 * 3 * 11)^2 + 4387*4

Now this equation, whatever modulus we reduce the right side to, it HAS to be a quadratic residue, else it will break the left side.

And ofcourse visa-versa.

So this reduces it to finding combinations on both sides that are quadratic residues in all mod p. 
And since this can be represented by legendre symbols... it can be reduced to a matrix in gf(2) ....

I cant see in my head why I cant solve it this way. I'm just so fucking depressed. These last few days, I literally feel like wanting to kill myself.
Guess being broke and hopeless about the future does that to someone.... it's just hard man... just so depressed it physically hurts.

I piss on everyone at Microsoft, except my former teamlead. They should burn for what they did they my former manager. And retribution will come, even if it takes me the rest of my life. I don't even care about getting harassed there, threatened with a gun, and then questioned like a fucking criminal who never belonged there and supossedly didn't do any work.. get my work visa revoked and seperated from  my friends, what they did to one of the only people in this industry who ever supported me is something I can never forgive. They will pay in blood. That entire fucking company will pay in blood. 

Anyway.... I have figured out how to represent this problem finally and solve it by using legendre symbols in a matrix in gf(2) .... its actually fairly simply once you understand these quadratic coefficients mod p properly ... and how they relate on both sides of the congruence.... this last week been shit... I know this is it, I know this is the tool I needed to finally do what I wanted to do.. to bypass the bottlenecks in v2. The time has almost come now... and with this, I will bring justice into this world again and make those bleed who cause nothing but ruin to the lifes around them. Without justice this world is shit, and I will not accept living in a shit world.

You know... I knew there was a way to do it. I know it was basically just looking for linear dependence, and that implied there had to be a mathematical tool to do this.. it just took me nearly two weeks to figure out the right tool.. but legendre symbols are perfect for the job.. it's literally all coming together like a match made in heaven. Having quadratic coefficients mod p... being able to use legendre... all the stuff I figured out in the last 2 years... it's been a chaotic journey... but this is it... I just know it.

If anyone is listening... and you want to stop what is going to happen.. happy to chat: big_polar_bear1@proton.me  ... ofcourse I will need to see justice for my former manager, because I rather burn this rotten world down then to live in these ruins you people made.

I will start writing it now... and it will come online as soon as its done.. if people cared, life wouldnt have been this nightmare..

ps: all the people simping for msft, I know and will remember all your names. Youre all the fucking enemy.

Fucking shit day. Some days I wish some piece of shit wannabe gangster tried to pick a fight with me, so I can fucking murder them. Maybe thats just the trauma of having been threatened with a gun and everything else that happened... but lately, I often feel like that. Lets just get this code done with this weekend... get this factoziation shit over with so I can focus on my revenge against Microsoft. For all care, Microsoft are as big of a pieces of shit as those fuckers who threatened me with a gun, they are the same. 

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

Holy fuck. It just dawned on me. So right now, we can already find a square relation mod m. Without effort. Instantly. Which number field sieve has to do alot of work for to achieve... but we do it instantly. The only thing that's missing is a quadratic character base ... to make sure its also square outside mod m.. which increases the likelyhood of it being square in the integers. And we're done. TIME TO END THIS. PREPARE YOURSELVES!

Its just number field sieve, but better... ergh tired. Let me continue figuring this out tomorrow. 

Number field sieve uses some really large prime field to take square roots over... I guess that is better then taking square roots over a composite modulus, a large primefield would only yield 2 roots for a given square. Hmm. Ok ok. I can do this, no sweat.
Feeling an incredible urgency suddenly to finish this now. I need some sleep first though.

Fuck fuck fuck, its number field sieve, but simplified, I need to adjust to a large primefield and then use tonelli to take the root. I believe that should fix that issue with finding that correct iN value.. i need some sleep first. brain is overactive. Fuck. I should have studied abstract algebra harder so  i could read these math papers on numberfield sieve, then it wouldnt have taken months to make this fucking connection. Someone else will beat me to publication if I dont hurry lol.
 DAMN IT, I DESPISE HOW BAD I AM AT MATH. fuck. Too fucking slow. I need to hurry now.

 After this, i swear, i will actually study math instead of trying to yolo it like this and losing my sanity.

im just gonna go be homeless somewhere far away from thus shit country. better then this life.

All this fucking nonstop bullshit. Money frozen, bamk accounts closed, legal harassment by Belgium. Never seeing any friends anymore. Im done man. My life is over. 

Fuck this math. Nobody even takes this shit serious, whats the point. my life is over. just been delaying what I should have done a long time ago.

Really bad day today. Really really bad day. My parents asking me about this justice department shit again, they dont know yet I told them I cant be bothered and that they should just go to court and that I'm leaving the country. All for sending an angry mail to the FBI. WHich I dont regret, bc America ruined my life. I should have never moved to the US. My life was great in Canada. They ruined everything I had. So yea, go to hell. I don't care. 
The way you finish this math is simple, you need to use some type of quadratic character base .... so when you combine things, we end up at the correct iN value.... because thats all this is.. you need to find some iN value, where each solution is a quadratic residue for any prime. So yea... I see the link with number field sieve now.... I'm just so stressed and depressed. I might legit flee the country and become homeless somewhere. I think thats probably what I'll end up doing. There's some videos on youtube about people being homeless and living in a tent... doesnt seem to be the worst life.

UPDATE: I ADDED A WAY TO POTENTIALLY FACTOR THE i-value (or iN value) to the paper. This seems to work, because if its not a factor of i, then it wont produce squares mod p if we follow the formula and increment the coefficient. So we can factor i without knowledge of i. In theory... don't think I'm missing anything? I'll write a PoC tomorrow... today is a really bad day... I feel an inch away from just going into the woods and hanging myself. Life has not been great. For a long time now... and I really miss my former friends.

Hmm, I guess even if a prime isnt a divisor of the i value, we can still figure out the i value by finding the correct offset it shifts by each increment of the coefficient.... something like that... and that should allow us to very quickly find that i value... in theory.. and once that is found, finding the factors of N is super trivial. It sounds so simple... I wonder if it can be true. I need a break for the rest of the day though... I'll write some code tomorrow... the stress of my living situation is really catching up now... the days I had my own appartment, near the beach in Vancouver... getting coffee with my teamlead... it seems like a far away dream, one I can never go back to again. I will end my life soon if this math doesn't work out... there is just nothing left in my life anymore. 

Maybe tomorrow.... this approach seems promising.. and I've arrived at a point now, where there's is very little left that I don't understand yet. Although, I wish I could spent sometime to study abstract algebra for a few months, stress-free, and revisit those number field papers so I can circle it back to my own findings... but I either push through it now, get results, or its over, my life is literally over if I don't get results right now. Anyway... figuring out this iN value... which I know now exactly the math behind... I think my approach has a good chance of working. At the very least it weakens the factorization problem to factoring that iN value... although I am fairly sure I can just figure out that iN mod p for any given prime... its just moving with a static offset mod p on each increment of the coefficient... it's not super high complexity... lets see tomorrow... time is truely running out now... because I will leave the country and go live in a tent somewhere, I don't care anymore... although once I get tired of living in a tent and being broke I'll probably just kill myself up in the mountains.

Hmmm... if this calculating of the correct iN value works like I think it may work... then tomorrow could be the day factorization comes tumbling down. Figuring out that correct iN value is basically congruent to factoring N. Lets see tomorrow. I have to hurry now, I am mere days away from the walls closing in on me. These people in Belgium want nothing more then for me to just vanish into some mental institute for the rest of my life... they nearly succeed when I was a kid, and losing years of my life in those places already and the insanity of them... I swore I would fight to the death before ever going inside one of those places again. I had nightmares for years. And I still intent to fight to the death if thats what people will push for. 

Tomorrow, I'll wake up and break factorization. I'll do it. I should know everything I need now. I'll break factorization and fix this world. I'll turn this entire world gay and queer as shit and break the minds of all the haters. 

Update: Wait wait, the conclusion I made that I added yesterday to the paper isn't correct. I mean, this number progression is correct:

1^2+4387\*65\*(1^2) = 534^2 <br/>
2^2+4387\*65\*(2^2) = (534\*2)^2 <br/> 
3^2+4387\*65\*(3^2) = (534\*3)^2 <br/>
4^2+4387\*65\*(4^2) = (534\*4)^2  <br/>
5^2+4387\*65\*(5^2) = (534\*5)^2 <br/>
6^2+4387\*65\*(6^2) = (534\*6)^2  <br/>

etc ..

Where 65 is the i value we must figure out somehow. 
There is some residue math here to calculate this.. I just know it... just got to get focused for a few days to figure it out.

God damnit man. This is just like number field sieve in a way... the solution is at the tip of my tongue now... an inch away from solving it. I just know it.

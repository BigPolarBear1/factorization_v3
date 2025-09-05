Uploaded 2d sieving. 


To build: python3 setup.py build_ext --inplace</br>
To run: python3 run_qs.py -base 4000 -keysize 200 -debug 1 -lin_size 100_000 -quad_size 100</br>

The math is working. Its mainly bottlenecking due to refreshing the modulus every 10 smooths and calling into sieve_quads... but I'm going to rework that logic when I get back from running. No point in calculating all of it if we refresh the modulus after 10 smooths (which is deteremined by the mod_refresh parameter... I'm not sure what the best value there is, I'll experiment later.. but for very big numbers when smooths become more rare, it won't matter anyway, it just slows down smaller numbers).

And when working with lists of primes there is also some tricks we can do to speed things up, since it's a unique list.

I also havn't yet implemented any tricks to gain an advantage from 2d sieving. But what I do already notice is that it is very effective at finding smooths. More so then just using one quadratic coefficient. It seems some quadratic coefficients just yield more smooths for certain moduli or something like that. Anyway... I need to optimize the code yet.

Then there is some additional stuff I want to explore... like really targeted smooth finding. I.e if we find a smooth candidate, we explicitliy go looking for factors with odd exponent in our 2d sieving setup. The math to do it is all there... I just need to figure out if it makes algorithmically sense.

Depression is getting worse and worse and worse. Plus there is a plane flying overhead at regular intervals that doesn't show up on flight radar. Pretty sure I have a hardware implant and they are collecting the data.. something like that. Whatever. Not like I'm going to connect my cache of 0days to this laptop anyway lol. I know how you people operate. I know and see every move you people make. Fucking dumbasses. 

If a visa can be arranged through the Chinese embassy, I am leaving this place. I know exactly what people are doing. 
Could have just apologized to my manager and paid him some money, like a year ago, and I would have been perfectly fine selling my work to the west.
But now a year later... we are in too deep, and I can't forgive this last year. I despise the west. I truely despise it. I despise belgium most of all. 

Update: back from running. I'll shower and get some food and make a few more edits to the code before I call it a day. I'm going really bad places mental health wise lately. I know that it was a similar situation where I did all my best logic bug work many years ago, just the same anguish and hopelessness. There seems to be a link between creativity and negative mental state. There probably is some genetic explanation. Because I feel like in the same mental state a starving animal might be in desperate search for food. Perhaps it creates some out-of-the-box thinking in a last ditch effort to prevent death. I got to keep pushing. One day I'll cross the threshold, and win. And then I'm off to go work in China... because fuck this shit. I know how they treat me here... while harassing me through the justice department for sending an angry email to the FBI. After all this, being treated like this after a long and succesful career and everything I did while at microsoft.. I truely despise western infosec. And I have a hatred for my own country that will never go away. I know what is going on. You people created this situatio you fucking pieces of shit. I told the FBI i was working on this. I told microsoft I was working on this. I fucking told everyone. And now you treat me like this. Go to hell. I'm off to China first chance I get (if it happens through official channels and not some anonymous guy emailing me because I'm way too paranoid). I want to go to China to fight the west. THat is how much I hate you people right now. And it will never change. 

Also, they killed the referal link shit in my github under insights.. you try to deny me my sigint visibility by making even more noise. You people suck at your job. You should quit. You are losing to a single polar bear. All you people should quit your job on the spot. If you don't, maybe you'll all start dropping death. Strange things happen. Fuck you. We're at war assholes. Make no mistakes. If you're part of the people moving against me, I promise you, endless misfortune will happen to you. You shouldn't mess with things you people cannot possibliy comprehend. Fucking losers.

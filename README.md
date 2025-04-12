# factorization_v3

Dropping this weekend. 

So if you have two squares mod N

Say N=4387

We can construct a relation like this: 4^2 = (4387-4)^2 mod N

However, taking the GCD on this ofcourse results in 1 and N.

However if we find a different root mod N for one of either square, then the GCD finds the actual factors.

I figured out I can use my number theoretical findings to do exactly that. It was right there infront of me all along..

Its literally just looking for a similar value mod n amongst the roots for a given squared coefficient combination. Which I know how to do, easy.

Enjoy the post-factorization world in a couple of days 🐻‍❄️✌️

Update: I think sunday I will be able to upload a  python PoC. After that I'll write a c++ version too, since its not a very complicated algorithm in its final form.
The final form doesn't use matrices, or fancy math... it's just number theory and quadratic roots and coefficients mod p. Depression levels are high today. Find myself often thinking back to my years in Vancouver, hanging out there with my former teamlead, running the sea wall, all the friends I had there. These big tech companies shouldn't give people visas if they intend to take it all away again after years. My entire life was in the pacific north west, all my friends, for years.. just to come back to this miserable place where I don't have anything anymore. Memories of better times... they will eat you alive until there's nothing left.

I guess its just the same like v2... but using the roots of our coefficients mod p to build the other square. I guess number field sieve made me realize that relation between roots and coefficient... should have realized it much earlier though... I'm depressed. I really miss my friends. And having turned 35 last month... you realize you are now entering middle age... not young anymore.. and life is really insanely short, and the realization that I'll likely never live and work near my friends again in my life... especially with the current politics.. I don't know. I don't know what I'm supposed to do. I think the hardest part to coming back to Europe is not being able to find employment and actually having people say law enforcement is threatening them with sanctions for even suggesting to buy my 0days. That coupled with just the complete social isolation and from time to time, people slinging transphobic slurs.. it's grinding away at my sanity every passing day. Writing this down on github at the end of every day, it's literally the only way I'm able to cope right now. And running 10k 5 times a week. Wish I could have gone on my arctic expedition... but my bank really screwed me over too by closing my account and freezing my investments, which made it unable to cash in on profits before the market crashed.

Update: Bleh, only slept 4 hours. Depression really fucks with your sleep schedule. Ever since I was threatened with a gun, fired from msft and forced to move bakc to Europe, my mental health has been really bad. I think its just depression, trauma and isolation... its hard some days. I remember hiking in the Arctic and having to walk 30km across mountain passes for 3 days without food.. those were 3 very difficult days.. but it also taught me that suffering is just in the head, and that you just need to keep going, no matter what. I could just take a rope, walk into the woods and hang myself.. but then what? Thats exactly what people want me to do. All these transphobic shitheads. All these fuckers in infosec who don't want someone like me to have a career. I'm not going to let pieces of shit like that win. Anyway... time to finish this code today. It's time to set the world on fire and burn it all down.

Update: Its kind of funny... seeing like expert math folks make these huge blunders in their PoCs... and not just on the programming side, on the math side too. I fixed a public number field sieve PoC, it wasn't doing at all what it was supposed to do.. in a matter of fact, it was finding squares of the form a^2 = (n-a)^2 mod n, and then calling into the sage library function sqrt() to factor n and find a different root... bc they did not  calculate algebraic smooth candidates properly. I mean, these things ofcourse happen.. my first LLL PoC missed the mark more then any of these PoCs. After 2 years... I'm finally starting to feel a little more confident in my math abilities. And my own work has truely come full circle... working on the final version as we speak... not sure if it will be done by tomorrow (depression just playing a number right now)... but definitely in the coming days. It all finally clicked, how roots and coefficients of quadratics mod p work together, and taking square roots over finite fields... and how to put all of that together to improve what I was doing in v2. This won't be another number field sieve PoC.. this will be something much better :).

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



# factorization_v3

Dropping this week.

Got the matrix math working now.

So one way of doing it... just factor what you can over the factor base. Then the remainder can be add to the matrix row as a jacobi symbols modulo p1, p2, p3, ... ,.. 
So that way we don't need to completely factor it over the factor base.
Then the hope is that the part that isn't factored over the factor base has similar factors when dealing with larger numbers. Or at the very least it allows us to find squares in our finite field... which can be useful bc then we can just take the square root in said finite field.. and hopefully that will then yield the correct square root. It's the best I can do at the moment I think. I wish I had access to a professional number theorist to bounce some ideas off.
I studied number field sieve again all day yesterday, and aside from doing the above, it's hard to do the exact same as number field sieve bc in our example we don't have polynomials with roots and coefficients congruent to N... we have tiny pieces of it in a way.. so it's completely different. And hence many of the tricks are kind of difficult to adjust to my own findings. I would need to do some further studying.

So anyway that works in my code. It will find the square relation while still being congruent mod N on both sides using those jacobi symbols.
I'll plug that into my v2 logic next.

Then after that I can see if it can be further improved by just taking the square root over a finite field incase the total product isn't a square in the integers.
The trick will be to have the result still congruent mod N on both sides... but I can figure that out, I'm sure.

I'll do that tomorrow. 
My head is just not in a great place. These financial issues are really just starting to give me a headache. 

This has to be finished this week. And then straight to earning money somehow. It's all becoming very dire. Even if its not 100% perfect, enhancing v2 with the jacobi symbols and roots over finite field should still be an improvement. 
Then focus on money, and in the meanwhile I should start porting it to c++ on the weekends.

I don't think anyone cares, but if someone wants to buy my work, big_polar_bear1@proton.me ... Ideally a contract with base salary and bonuses for succesful research would be ideal. I really need some stability so I can have a normal life and be independent. I can go back to vuln research too. I guess my math work is a work in progress that will likely take a few more years to really mature. I mean.. math phd's or even masters they have like half a decade of dedicated math education, if not more. I have just been doing it by myself with the help of youtube vids and books for 2 years. Although I am hoping that hyperspecializing into one problem can give me an edge... same strategy I used with vuln research. The hardest part about math is however, it is 1000 times more demanding then vuln research, and half the time, while doing it I feel on the edge of a mental breakdown.. but at the same time.. I love it more then anything else. Until I die, I don't think I will ever let this go, even if I have to relegate it to the weekends for a little while. I know there isn't many actual math folks who think highly of my work.. but it is something I must do regardless, it's the same compulsion that got me into vuln research.. it's how I'm wired I guess. Number theory for me, is what I wanted vuln research to be... there is just endless depth to it.. one can easily get lost.

Update: Time to finish the general outline in code today. Then spent a few more days refining it and publish v3 this week. I am so broke, I cant deal with my financial situation anymore. I am at risk of going into debt now. Especially with all this shit from my bank they are doing. I keep callimg them and making appointments but nobody knows anything. Literally wondwr some times if this is just some shithead in Belgian law enforcement who really doesnt like me.. bc living in Belgian is a fucking nightmare, things just always seem to work against me here. Never had these issues in Canada or the US.

You know, if you really think about it. Just entertaining science fiction, perhaps the singularity would be the only way to save humanity. It is true that such super computer would not have any humanity. But I truely believe all human cruelty has trauma at its roots. A never ending cycle of violence and trauma, causing more violence and trauma. Perhaps could the living standards for all humans be raised so dramatically, as to reduce trauma, it may just solve human cruelty and free us to pursue the things in this world that really matter (understanding and seeing the world we live in and ourselves). Thus even if a super computer guided human society, perhaps it would solve human cruelty, despite its lack of humanity and finally free us from this never ending cycle of violence, so we can finally pursue our greatness. Anyway, time to get to work. Enough wandering of the mind for today.

Update: I did it! It works. This updated version of v2 can now find factors using smooths that don't factor completely over the factor base by using legendre/jacobi symbols. Big personal breakthrough. Next, incase we don't achieve a square in the integers, I should take the square root over the finite field.. since if it matches jacobi symbols for each prime in our finite field, it also means it must be square in the finite field. My only worry is maintaining congruence mod N on both sides I guess. We'll see how to sort that out.


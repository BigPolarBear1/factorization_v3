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


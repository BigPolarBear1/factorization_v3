Uploaded QSv3_056_2d_sieving_WIP.py

To use: pypy3 QSv3_056_2d_sieving_WIP.py -base 500 -keysize 100

TO use the old PoC (that one will easily factor above 200 bit):

pypy3 QSv3_050.py -base 6000 -keysize 200

Made some more improvements to 2d sieving. The biggest bottleneck right now is constructing the 2d sieve interval.
So it is time now to optimize that. In theory instead of constructing row by row, we should be able to fill out the interval in 2d dimension. Hence having to execute a whole less code.
Additionally this would then be perfect to be further optimized with SIMD I guess.
Let me start optimizing that code.

Because each process is now responsible for a range of quadratic coefficients, the process at index 0 (which has the smallest quadratic coefficient range) will yield the most smooth. Hence I think it is better to move parallelization to the sieving process itself instead.
There is still a lot of work to do with this 2d sieving. But I am feeling extremely optimistic about this approach (it may not seem like it yet, but just wait... you will see soon! I know what I'm doing now). I should also fix the paper soon, so that once the PoC is ready I can make some noise about it. It's definitely wildly different from default SIQS.

Anyway, I'll go for a run. Figure out the math after that to optimize constructing that 2d sieving interval and then tomorrow implement it in code... and then we should slowly start seeing the true strength of this approach. This really should be many times faster then normal SIQS and not even talking about its potential to be optimized further with SIMD.

Actually let me add simd tomorrow with cython. Easy enough. :)

I jsut realized, I really can move all of the linear congruences to the initialization phase of the algorithm, out side of all the loops. Thats the beauty of working with moduli like this. A shitload can just be precomputed. And I really need to start using that as a strength. Then the heavy lifting that remains in the algorithm can be hyper optimized with things such as SIMD. Yea, this is going to be beautiful once its done. I was doing some more reading, I cant find anything similar on the public domain, I did read about lattice sieving in number field sieve, which may be similar but I'm not sure but either way, doing something similar in quadratic sieve is a good thing. I don't think default SIQS could achieve this, because they arn't working with moduli and without that reduction things get complicated very quickly... still... I really want to finish this and move back to a deeper study of some of the stuff I was doing before. It is really infuriating bc I know I am right, I know I am on to something... yet I am completely ignored. I'll wrap this up and start aggressively sharing it online I guess. It's my only hope for a better future. 

Update: Bleh, super low focus day. And tomorrow I want to attempt running marathon distance for the first time in my life. Guess I'll just take a break from work this weekend and resume on monday. If I don't die while attempting a marathon with this heat. You know, I really want to finish this v3 version asap. Hoping to be done by september so I can pivot back to v4... which is really the approach that gets me most excited.. it feels like deep space exploration, pushing off into the unknown... there is something there, a hidden monster lurking beneath the depths... I need to figure it out even if it takes the rest of my life. I don't care anymore if people laugh at my work, I'm not doing this for other people, just because I have to, for myself. I just can't let go of this anymore. It has taken over my life for over 2 years now... and it's just pulling me in more and more... and it's not because of having invested so much already... it's just that the deeper I get, the more I begin to see the outline of something amazing.. and it's still just this vague picture, I have to figure it out, I have to see it. People don't understand this, they all just think I have become psychotic and need to be in a mental hospital or something. Life is too short to waste it on things that don't matter to yourself. I rather end up homeless and broke if it means I can keep working on this. I'll work on this until my last dying breath.

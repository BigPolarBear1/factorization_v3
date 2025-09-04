To build (from the PoC_Files folder): python3 setup.py build_ext --inplace</br></br>
To run: (note the actual factor base per quadratic coefficient will be about +/- half of the -base parameter and currently worker support is also not working)

140 bit: python3 run_qs.py -keysize 140 -base 1000 -debug 1 -lin_size 100_000  -quad_size 1 (2 seconds)    </br>
160 bit: python3 run_qs.py -keysize 160 -base 1000 -debug 1 -lin_size 100_000  -quad_size 1 (8 seconds)    </br>
180 bit: python3 run_qs.py -keysize 180 -base 2000 -debug 1 -lin_size 100_000  -quad_size 1 (16 seconds)   </br>
200 bit: python3 run_qs.py -keysize 200 -base 6000 -debug 1 -lin_size 100_000  -quad_size 1 (100 seconds) </br>

Note: I am now finding myself doing functionally the same as SIQS with just more overhead to calculate the factor_base. Which is not good. I will explore the 2d sieving part again and also lifting, but I need to move away to v4 asap, bc I'm burning too much time here.

I'm so depressed. 

To get any strength out of this number theory compared to normal SIQS, you need to achieve 2d sieving. Without that it is pointless, then you just end up doing slower SIQS. 
Looking at this stack overflow PoC, seeing this trick they do to enumerate root values for multiple linear coefficients within the same quadratic coefficient... I think I can do the same by enumerating quadratic coefficients within a modulus.  I think some approach like that is how you succesfully do 2d sieving. Let me bash my head against that today.

I'm so awfully depressed. Today is really bad. Having a lot of suicidal thoughts. 

That is it. If you know one quadratic coefficient in mod m, then you can very quickly calculate all the other ones with no overhead.
And we can very quickly calculate its linear coefficients with little to no overhead. 
And that is how you construct your 2d sieve interval with little to no overhead. 
I mean, the math is all there and that stack overflow PoC shows me how to do those calculations. 

Ok, 2d sieving or death. Lets make it happen. One final effort. 

Update: So i've added the trick to very quickly calculating linear coefficient enumerations using precalculated constants. Then we can do the same to get the dist1 and dist2 variables for creating the sieve interval... but using purely coefficients that looks a little different.. I think it uses different constant based on the legendre symbol... I need to manually run the numbers to figure out what the constants should be. That, I hope to finish today. 

Then tomorrow I will try to port that same trick to quadratic coefficients.... and somehow get the math right to quickly perform 2d sieving in a way that actually gives us an advantage, if it all possible. I'm struggling with brain fog due to depression a lot.. have been for some months. Life is not great.

My initial intuition and appraoch with the residue map wasn't bad. But I need to work with one single modulus. This how you can apply a lot of cool tricks. And I need to actually think about the design careful. Like pen and paper, and try to not be depressed and find some creativity. 2d sieving. I know it can be done. Constructing all permutations of quadratic coefficients using that precomputed constants trick is easy, no overhead there. Then every generated quadratic coefficient will have a unique pair of linear coefficients mod p. Here we apply the same constants trick again to generate all permutations without overhead. For the distances in the sieve interval mod p, we again use that constants trick to enumerate them.. I nearly have that math figured out in code... just one bug I need to fix tomorrow... and we can save the "seen" residues mod p and re-use them if we hit the same residue again. But without precomputing residues like we did before that's too slow.... just do it all on the fly. Sounds good... time to get too work. Almost 2.5 years ago when I started with 0 math knowledge, as a highschool dropout.. I do not care what people think. 2.5 years is still less time spent on math then the average undergraduate. I will keep pushing, I will never give up. Even if I decide to go slower to make some money in the future, I will keep working on this. I will run out the clock, and eventually I will succeed. I may not be as smart and sharp as some people, but I have determination, creativity and trauma... hahahaha. As long as I'm making progress, as long as I still have ideas, why would I stop? Life is short, I have one chance to do something truely unique, and I'm crazy enough to give it a try at the cost of everything else. I don't care about the rest.

I'm also wondering if I should not just completely step away from QS. Because with those findings, you can reduce the problem to finding a small linear coefficient with a large modulus (this increases the likelyhood of smoothness). I'll mess around some more with that idea for v4 I guess. Tomorrow implementing the 2d sieving the proper way shouldn't take much time... we'll find out quickly if it is feasible or not. Else I think I'll just trash this version and move straight to v4.

Update: I just keep getting these intrusive thoughts that keep me awake at the moment. So lets say we take '2' as linear coefficient. .. we figure out in which moduli it appears. Then using this constants trick we can very quickly check if there exists a small quadratic coefficient and test for smoothness. I don't know..... fuck it, before I do anything else, tomorrow I need to write some code to try this. Just incase this was the obvious thing I missed. I mean... this idea has been grinding in the back of my head for a while... but now that I know that  constants trick to  enumerate linear coefficient combinations quickly...this may be what I need to pull it off. I have to try it tomorrow otherwise it won't let me go.

SLKDJLAJDLSKDJL:A:SADIOJ erggggggh. Fuck. Why do these ideas come at 3am. It sounds like a good idea. If we have a small linear coefficient and large modulus, then all we need is find quadratic coefficient such that when we calculate the discriminant and subtract from its bit length the bit length of the modulus, we are below a pre-set bitlength (determined by the size of the factor base). And we don't need to calculate the discriminant each time, if we know the bitlength of N and the bitlegnth of our quadratic coefficient, we just add them together. Hence we only need to check the bitlength of the quadratic coefficient. Then the only bottleneck would be, how often do small enough quadratic coefficients occure? Or are we going to end up with a deal where we have exponential growth? Hmm. Now... a real math research would probably already know the answer, but for an amateur like me, I will go write some code, and find out myself... first I need some sleep. 

Never give up. I succeed or I go into the woods and hang myself. THat's all that is left. A fight to the death. I don't care anymore. Shouldn't have fired my manager. 

Update: Or wait that constants trick only works with linear coefficients. So we need to use that constants trick to iterate linear coefficients for a given quadratic coefficient. And find a linear coefficient below a certain threshold. I wonder... like take quadratic coefficient 1, thus N\*4\*1 .. then grab a huuuuuge modulus. Calculate an initial linear coefficient and use that constants trick to spit out permutations of it quickly. We don't do sieve intervals... just look for one that is small enough (I wonder what the incidence of "small" enough coefficients will be?). I like the idea. It's similar to what I did in v2 ... but with what I know today, I think I can do it much better. I should give it a try atleast.

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

Update: Yea, I need to zero in on 2d sieving for now. I have a bunch of other ideas, but they will take a lot of further research. I also see some connection to number field sieve's approach... but again... I need time and I'm depressed, I need something tangible asap.. so 2d sieving... lets go.


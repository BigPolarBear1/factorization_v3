Update: One thing I must check for sure is, if we pull all smooths from a single modulus. Can we take square roots over the modulus like number field sieve does? .. or find some connection to number field sieve... I will quickly have a look at this next.  Because that could also dramatically reduce the number of required smooths... even if we are forced to work with a single modulus. Remembering last time I messed around with this, not having all my smooths in the same modulus was a headache... this may actually solve it now. Then we have succesfully merged quadratic sieve and number field sieve. So fucking depressed. This fucking last year man. How they treat me. Like I don't see it. 

Uploaded 2d sieving. 


To build: python3 setup.py build_ext --inplace</br>
To run: python3 run_qs.py -base 6000 -keysize 200 -debug 1 -lin_size 1_000_000 -quad_size 100</br>  

Bah, I actually did some debugging, and found out that you shouldn't use even quadratic coefficients because those will generally yield bad smooths.
Let me optimize the code next. The 2d sieving is working, but it is still wildly unoptimized and we arn't using any tricks yet to gain an advantage there.

I have a bunch in mind. Plus we also need to speed up calculating sieve interval distances for permutations of linear coefficients (using that same constants trick we use to calculate permutaitons of linear coefficients).

What I really want to explore is targeted smooth finding. Factor a smooth candidate and then go hunting for other smooths that contain its factors with odd exponents, in the hopes of being succesful with way less smooths. 
If I can get an aproach like that working.. that could be beneficial. And I'll have to think some more how to possibly get advantages out of 2d sieving.

Update: I'm doing some experiments right now. Just pulling smooths from as few moduli as posible (meaning all smooths will have more similar factors) seems to massively reduce the amount of smooths required to be succesful. I think this approach is the way to go. I will zero in hard on this now. Because if I can get this to work really really well... then we can potentially factor very big numbers requiring only a fraction of smooths. How cool would that be?
So what I think we need to do is, just create a sieve interval for one quadratic coefficient, then when we process the sieve interval and we find a smooth, we go looking at other quadratic coefficients to find similar factorizations (not just the modulus since that represents only a portion of the smooth factors, but also the other factors). And then we can change some logic, like every 100 smooths, go to the linear algebra step, and if it fails, continue finding more smooths. Something like that. That would be quite awesome if I can get that to work the way I think in my head it might work. Alright. Got to stay focused. Got to stay optimistic. This will end soon, one way or another.
Plus p-adic lifting can also result in more similar factors. hmm. Oh... there is actually many ways to lower the required amount of smooths... this may actually by a valid appraoch..... yea yea. We can do this. Time to destroy america's cryptologic advantage. Fuck you you ugly transphobes.

Update: Made some more changes. It will now go to the linear algebra step every 500 smooths. Next we will bring the number way down where it succeed at the linear algebra step... because I know exactly how to do it now thanks to my 2d appraoch. Fuck you america. Fuck you for all the trauma. Fuck you for firing my manager. Fuck you for the last year. Fuck you 10000 times over. Fuck pete hegseth, fucking coward piece of shit. Fuck you all. I'll fight you till my last dying beath. I'm not afraid. You killed me 2 years ago. I'm a ghost here to haunt you you stupid fuckers.

Update: I did some quick testing, and deliberatly hunting smooths with similar factorization (after the modulus is subtracted) it does dramatically decrease the required amount of smooths. So yes, that is how you properly leverage my findings. I'm going to need a few days to implement targeted smooth hunting... and then another few days to also add p-adic lifitng to further reduce the amount of required smooths. Hopefully to the point where only a few smooths will yield a succesful factorization. That would be cool.



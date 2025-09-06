Uploaded 2d sieving. 


To build: python3 setup.py build_ext --inplace</br>
To run: python3 run_qs.py -base 4000 -keysize 200 -debug 1 -lin_size 100_000 -quad_size 1000</br>  

Bah, I actually did some debugging, and found out that you shouldn't use even quadratic coefficients because those will generally yield bad smooths.
Let me optimize the code next. The 2d sieving is working, but it is still wildly unoptimized and we arn't using any tricks yet to gain an advantage there.

I have a bunch in mind. Plus we also need to speed up calculating sieve interval distances for permutations of linear coefficients (using that same constants trick we use to calculate permutaitons of linear coefficients).

What I really want to explore is targeted smooth finding. Factor a smooth candidate and then go hunting for other smooths that contain its factors with odd exponents, in the hopes of being succesful with way less smooths. 
If I can get an aproach like that working.. that could be beneficial. And I'll have to think some more how to possibly get advantages out of 2d sieving.

Update: I'm doing some experiments right now. Just pulling smooths from as few moduli as posible (meaning all smooths will have more similar factors) seems to massively reduce the amount of smooths required to be succesful. I think this approach is the way to go. I will zero in hard on this now. Because if I can get this to work really really well... then we can potentially factor very big numbers requiring only a fraction of smooths. How cool would that be?
So what I think we need to do is, just create a sieve interval for one quadratic coefficient, then when we process the sieve interval and we find a smooth, we go looking at other quadratic coefficients to find similar factorizations (not just the modulus since that represents only a portion of the smooth factors, but also the other factors). And then we can change some logic, like every 100 smooths, go to the linear algebra step, and if it fails, continue finding more smooths. Something like that. That would be quite awesome if I can get that to work the way I think in my head it might work. Alright. Got to stay focused. Got to stay optimistic. This will end soon, one way or another.
Plus p-adic lifting can also result in more similar factors. hmm. Oh... there is actually many ways to lower the required amount of smooths... this may actually by a valid appraoch..... yea yea. We can do this. Time to destroy america's cryptologic advantage. Fuck you you ugly transphobes.

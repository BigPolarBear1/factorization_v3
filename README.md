Uploaded 2d sieving. 


To build: python3 setup.py build_ext --inplace</br>
To run: python3 run_qs.py -base 4000 -keysize 200 -debug 1 -lin_size 100_000 -quad_size 1000</br>  

Bah, I actually did some debugging, and found out that you shouldn't use even quadratic coefficients because those will generally yield bad smooths.
Let me optimize the code next. The 2d sieving is working, but it is still wildly unoptimized and we arn't using any tricks yet to gain an advantage there.

I have a bunch in mind. Plus we alos need to speed up calculating sieve interval distances for permutations of linear coefficients (using that same constants trick we use to calculate permutaitons of linear coefficients).

What I really want to explore is targeted smooth finding. Factor a smooth candidate and then go hunting for other smooths that contain its factors with odd exponents, in the hopes of being succesful with way less smooths. 
If I can get an aproach like that working.. that could be beneficial. And I'll have to think some more how to possibly get advantages out of 2d sieving.

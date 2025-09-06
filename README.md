Update: One thing I must check for sure is, if we pull all smooths from a single modulus. Can we take square roots over the modulus like number field sieve does? .. or find some connection to number field sieve... I will quickly have a look at this next.  Because that could also dramatically reduce the number of required smooths... even if we are forced to work with a single modulus. Remembering last time I messed around with this, not having all my smooths in the same modulus was a headache... this may actually solve it now. Then we have succesfully merged quadratic sieve and number field sieve. So fucking depressed. This fucking last year man. How they treat me. Like I don't see it.  We might actually be able to jump over to number field sieve's approach now... this may actually complete the puzzle now..... yea fuck you all. Go to hell for all you did. Fuck america too. I see your government constantly crying about trans people. I know who is my enemy and who is not. You should respect people. Yea, maybe learn about that you fucking losers.

Uploaded 2d sieving. 


To build: python3 setup.py build_ext --inplace</br>
To run: python3 run_qs.py -base 6000 -keysize 200 -debug 1 -lin_size 1_000_000 -quad_size 100</br>  

Bah, I actually did some debugging, and found out that you shouldn't use even quadratic coefficients because those will generally yield bad smooths.
Let me optimize the code next. The 2d sieving is working, but it is still wildly unoptimized and we arn't using any tricks yet to gain an advantage there.

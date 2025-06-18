I GOT IT!

Use: QSv3_007.py -keysize 50

Just added v007. Just some minor improvements over v6. 
As you see, when it finds a good relative i-value, we are almost garantueed to find a good coefficients (although we may need to add the modulus a couple of times).
The main bottleneck now is reducing the cartesian product of coefficients mod p<sub>i</sub>.
Which I know how to do... so expect it to be fixed shortly.
With that bottleneck removed soon, we'll see how fast it ends up being. After that we can gain some more performance here and there.. there is definitely a lot of i-values we can eliminate from the start which don't have valid solutions... so I may need to think how I'm going to do that in code... Lets see.
When all is said and done, I am hoping to overtake v2 in performance. Without having to find smooth numbers, which would be a major breakthrough. Plus all of this is in python... this has a lot of potential to be super-optimized since we only use straightforward operations.

# factorization_v3

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WORK IN PROGRESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Useage: python3 QSv3_version_c.py -keysize 30

Just upload version_c (31 may 2025).
It's still very slow, nothing has been optimized. But this will be the foundation for future versions.
The approach is as following right now:

Given a root and coefficient x<sup>2</sup> - x*y = a  we can use a hashmap to find another root for the same coefficient such that  x<sub>0</sub><sup>2</sup> - x<sub>0</sub>y = (x<sub>1</sub><sup>2</sup> - x<sub>1</sub>y) - XN  (XN = multiple of N)
I have to improve a lot of the parts of this code. I'll push updates regulary now. The performance ceiling for this should be fairly good once everything is completed. 
Going to take a break for the rest of the day. I'll push version_d tomorrow at the end of the day I think.

For the final version the goal is to query some hashmap datastructure and find a pair of roots that when used in a quadratic yield the same results mod N. Right now there is still too much brute force going on. Ideally we don't want to bruteforce roots and then try to find matching roots. We want to pull pairs of roots out of that hashmap without bruteforcing. It will get done eventually... give me a few days.
The upside of this approach is that we completely bypass smooth finding :). 

ps: I am absolutely aware its still slow. However, its about the math we are doing. With everything I learned while making v2... v3 is going to start improving really fast now that Ive written the "core idea". I need to make some adjustments to the code later, such that we improve the likely hood of hitting something thats the same value mod N. I know how to do it from v2... just tired right now. Expect all these things to get fixed in the next few days. This is basically going to be v2 without the need for smooth finding once all is said and done.

pps: For tomorrow, I'll direct my main energy to pulling roots from the hashmap such that we have the same result mod N. Similar to how it was done in v2, but its slightly more complicated since we arn't working with the squares themselves.. but I'm fairly sure I know how to achieve it. I didn't have the pieces to do it when working purely with the coefficients, but I believe I do have them now.

# factorization_v3

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WORK IN PROGRESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Useage: python3 QSv3_version_b.py -keysize 30

Uploaded QSv3_version_b

This is even slower then the previous v3 version. I know that!!
However, I am finally able to make the link to number field sieve. That's what this upload is about.

This will calculate coefficient combinations and roots, and then try to find another root that yields the same results mod N when solving the quadratic x<sup>2</sup> + xy = a mod N.
If it is the same, we can add or subtract both roots, solve the quadratic again, and use that result to take the GCD with N.

Now, we can truely begin. version_b just does bruteforce. But you don't have to do that. I'm not going to spill the beans completely yet (its really simple.. you can probably figure it out).. but version_c will solve this, hope to upload today or tomorrow.

hint: There is a very similar approach to how we created a hashmap in v2, to incorporate it with this. Almost done working out the details... I'll probably upload it tomorrow, going for a run and try to sleep after. This will all make sense within the next few days now... almost, so close.

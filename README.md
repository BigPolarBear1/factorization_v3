# factorization_v3

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WORK IN PROGRESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Useage: python3 QSv3_version_b.py -keysize 30

Uploaded QSv3_version_b

This is even slower then the previous v3 version. I know that!!
However, I am finally able to make the link to number field sieve. That's what this upload is about.

This will calculate coefficient combinations and roots, and then try to find another root that yields the same results mod N when solving the quadratic x<sup>2</sup> + xy = a mod N.
If it is the same, we can add or subtract both roots, solve the quadratic again, and use that result to take the GCD with N.

Now, we can truely begin. version_b just does bruteforce. But you don't have to do that. I'm not going to spill the beans completely yet (its really simple.. you can probably figure it out).. but version_c will solve this, hope to upload today or tomorrow.

hint: There is a very similar approach to how we created a hashmap in v2, to incorporate it with this. Almost done working out the details... I'll probably upload it tomorrow, going for a run and try to sleep after. This will all make sense within the next few days now... almost, so close. There's something interesting if you use partial results and plug them into that quadratic :) .... hehehehehehehehehehehehehehehhehe. I wonder if someone already knows... because then they know shit is about to hit the fan.

ps: I know the uploaded code is shit. Its not supposed to be fast. Its just bruteforce. But it shows that I figured out how the roots fit into the picture. When I figured out the other coefficient is calculated using the derivative of the coefficient and root.. thats when things finally started falling into place.. now one more thing remains.. doing all of this mod p, creating sometype of fast lookup like we did with the hashmap in v2... and then factorization will go brrrrrrrrrrr \*insert a-10 warthog sound\*
I'll do it tomorrow... pretty sure my work will be done tomorrow.. or atleast a first good outline of what the final version will look like. Almost there now. I know no-one believes in me.. but they just don't understand this.

Update 30 may: Lazy day today. Trouble focusing. Will probably upload a more final version this weekend instead of today. Going to wear dresses and woman's clothing only going forward. Tired of this repressed life. If people harass me, I'll fight them and do glorious battle hahahahaa. Fuck them all. Fuck this world. Anyway, the exact same type of xN hashmap as we did in v2 can be used here. I just worked out the math. It is 100% possible. I have a feeling people are about to have a really bad time very soon... I don't care anymore. Life was good, I had friends, my own place... and people destroyed it all. I call what is about to happen, polar bear justice. 

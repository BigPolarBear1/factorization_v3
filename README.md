Note: I know I am right. I know people know I am right. I know how you people treat me, despite being right. I do not care about the reasons. I will not forgive you people for stealing more then a year from my life where I had to struggle like this. I especially wont forgive belgium. I would renounce my citizenship and move elsewhere for this. I cannot ever forgive this. I've been broke for a year, I can't find work, I can't find 0day buyers, I can't see my friends anymore and I'm stuck, there is no dignity in living like this. A human life is very short, and you have wasted a good chunk of mine. There absolutely won't be forgiveness. We're past that now. And if I get the oppurtunity to work in Asia, I will go and there is absolutely nothing that can be changed about that. Because what you people are doing, I have no words for it, this is betrayal. Betrayal to the long career I had, helping find security issues in software like OpenSsl and other very important components.. you better hope this was my last discovery because next time I will remember who treated me like shit and who didn't. Remember that. The only way I may even remotely change my mind is if Microsoft publically apologizes to my former manager and pays him many millions. Without that, forget it, you'll have to kill me I guess.


Uploaded QSv3_simd.pyx v001

To build (from the PoC_Files folder): python3 setup.py build_ext --inplace</br>
To run: python3 run_qs.py -base 1000 -keysize 140 -debug 1 -lin_size 100_000 -quad_size 20

To use the old PoC (that one will easily factor above 200 bit using pypy3):

pypy3 QSv3_050.py -base 6000 -keysize 200

The simd version is a work in progress. Just uploading since I'm taking a break for the day and running a marathon tomorrow (so wont get work done).
This version demonstrates the direction I'm working towards very well.

There's many things still to do:

1. I absolutely need to process each row for smooths while building the 2d interval, so we do not need to keep all rows in memory since it limits the size of the 2d sieve interval too much right now (plus I need to switch away from native python types). For the future I have some ideas on how to use that 2d interval to do some post-processing to find even more smooths... but then I'll have to write it to disk since it gets too big to keep in memory.

2. The create_residue_map function is very slow. Speeding this up will allow for a greater height in the 2d sieve interval. I need to rethink this function and use numpy arrays

3. Make sure to use the numpy c api (cnp) instead of the python numpy api, this removes some extra abstraction. In addition move to numpy and c types wherever I can. We should absolutely not be mixing numpy and native python types, since this ends up an order of magnitude slower then even just using natie python types.

4. Add more simd support and use static typing.

5. One big thing I also need to address is building the iN map... because being able to precompute this quickly is one of the main advantages of my number theory. Right now it still takes too long.

6. Change how parallezation is done. The way I'm doing now eats away too much ram and is extremely suboptimal. I.e things such as the partial smooth relation hashmap arn't shared amongst workers.

7. I really really really need to implement lifting aswell. Because if we cap our quadratic coefficient to a range from, for example, 1 to 10_000, then working within that limit we can fairly quickly calculate p-adic linear coefficinet solutions. Which in turn also allows us to work efficiently with a much smaller factor base... which then also speed up other areas of the algorithm such as testing for smooths. I think this may be a key ingredient... but let me address some of the more urgent items on this list first.

8. etc etc.....

It will be really fast once everything is done. The implementation right now is still kind of sloppy... but anyway, if I die tomorrow from running, you can do it yourself too now.

Note: If you have a quadratic coefficient and a linear coefficient in mod m. You can exactly calculate the amount of bits your smooth will be after dividing by the modulus. Hence small coefficients and a large modulus are more likely to yield smooths.
There may be a way to pull these from our precalculated datastructure efficiently... but I will investigate this idea some more in v4. I think for v3, this will be good for now

ps: I absolutely hate having to work with cython, because to be honest, just working in pure c++ is less of a hassle... the only upside of python is that moving things around and refactoring is very fast and pain free... so once I am happy with the final shape of the algorithm (for v3 atleast) I may or may not rewrite it in pure c++... but I am hoping I can push it past 300 bit just doing this approach.. we'll see... still an awful lot of bottlenecks and frankenstein code due to my incremental way of doing research that I need to address... just depressed I guess, moving at a slow pace.. but I hope next week now I'll make some considerable progress. Once it's past 300 bit I can make some noise about it.. ideally I'll get it to surpass msieve's performance.. because then the proof is in the pudding. We'll see. Just kind of hate this phase of research... messing around with numbers is more fun then figuring out the details of how to implement an algorithm. I just really dont like computer shit anymore. I just hate computer shit.

Update: Shit day today. I ran my second marathon yesterday. This time I ran in a nature reserve, but running a marathon on sandy trials added a whole other dimension of difficulty. On the bright side, I was forced to slow down a lot, so that kept my heart rate well within zone 2 for the entire thing. I'm going to run 50k next sunday. Fuck it. Lets see how far I can push it until I drop dead. I don't care anymore. I'm just going to keep pushing the distance on my long runs until I drop dead. Fuck this shit. Fuck this shit world with it fucking shit people. I stopped caring at this point. Too much has happened, and now I have found myself completely stuck in life. I can still run, so that's what I'll do.

You know, it has been two years ago, since those things happend in the US. And I still find myself unable to cope with the mental scars. I now run 80+ km a week, purely because it is the only way for me to endure the mental pain which is a direct results of what happened in the US. I think about it alot, over and over again, and really, the world is a disgusting place. I wasn't optimistic about the world before... but now... the way I see the world now, I feel like it is making me lose my mind. It is unbareable and the only thing I can do, is run. Although ideally I would be alone in the Arctic right now, but since I'm broke, running will have to do. One day in the future, peope will look back to these barbaric times and wonder how we didn't all nuke ourselves to exstinction.... the entire world is unravelling... and I know if it wasn't, neither would those things have happened in the US that happened. The blatant cruelty, hidden behind thinly veiled lies and disinformation.. it will destroy this world if nothing is done. It's like the 1930 all over again, but with a lot more at stake this time around. People don't realize how easily it is for them to be brainwashed... ofcourse having to admit this, would be the same as admitting they have very little agency in this world. 

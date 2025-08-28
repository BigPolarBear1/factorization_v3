Note: I know I am right. I know people know I am right. I know how you people treat me, despite being right. I do not care about the reasons. I will not forgive you people for stealing more then a year from my life where I had to struggle like this. I especially wont forgive belgium. I would renounce my citizenship and move elsewhere for this. I cannot ever forgive this. I've been broke for a year, I can't find work, I can't find 0day buyers, I can't see my friends anymore and I'm stuck, there is no dignity in living like this. A human life is very short, and you have wasted a good chunk of mine. There absolutely won't be forgiveness. We're past that now. And if I get the oppurtunity to work in Asia, I will go and there is absolutely nothing that can be changed about that. Because what you people are doing, I have no words for it, this is betrayal. Betrayal to the long career I had, helping find security issues in software like OpenSsl and other very important components.. you better hope this was my last discovery because next time I will remember who treated me like shit and who didn't. Remember that. The only way I may even remotely change my mind is if Microsoft publically apologizes to my former manager and pays him many millions. Without that, forget it, you'll have to kill me I guess.

Update 28 August 2025

Uploaded QSv3_simd.pyx 

To build (from the PoC_Files folder): python3 setup.py build_ext --inplace</br>
To run: python3 run_qs.py -base 2000 -keysize 150 -debug 1 -lin_size 100_000 -quad_size 5

To use the old PoC (that one will easily factor above 200 bit using pypy3):

pypy3 QSv3_050.py -base 6000 -keysize 200

I've further optimized some things. I also tested with another PoC, I seem to be only getting half the amount of smooths, so I think I need to check my sieve interval.. or perhaps sieve in the negative direction. Let me think what's going on.
It's awfully slow. Hell, it's slower then the old version. This is purely because we've added a bunch more indexing and unless we use typed memory views or similar, indexing will be really really really agonizingly slow. So that's what's happening with that.
That will be the next thing I'll start to address now.. but it should be good once everything is optimized.. since again... we don't need to keep recalculating roots and polynomials. I just need to get rid of all the crappy python abstractions inside my main loop.

Oh and ofcourse the sieve row... that one shouldn't be the full length of lin_sieve_size... only about the amount SIMD can do at once. But I need to do some testing to make sure the code still ends up SIMD optimized.. I may have to learn cpu intrinsics eventually.. but SIMD or not.. just working on contiguous memory is much more important.

I'm also gonna do some refactoring. Less code inside that loop is better. That's the main thing. So get rid of create_residue_map. Just precompute those sieve_rows with a legnth of 256 or 512 (depending on how much SIMD uses).. that way memory isn't an issue... and makes that entire create_residue_map function unneeded. The main thing is, I need to throw as much code out of my main loop as humanly possible. That will by far give the biggest boost. I'll upload a better version tomorrow... going for a run now.

God damn, paralyzed by all this trauma. When I go outside running, just this insane hypervigilance. I need to finish my work. Figure out a way to earn some money so I can take some vacation for a while. Let me go take a shower and make some food, then just remove things from that main loop in my code. Just move as many lines of code outside of that loop as I can. I'm still trying to figure out what works and what doesn't, but from just messing around, nothing matters more then reducing lines of code... so let me give that a try. 

The worst part about trauma. is how exhausting it is. And some days I feel bad, because even when people are trying to be friendly, I just feel like I'm a bear who is about to tear someone to shreds. And I just cant help it. It's why I like being alone in nature, it's the only time I can relax. I wont ever forgive Microsoft. AT one of the most traumatic moments in my life, they decided to make everything so much worse. I won't ever forgive them and they will pay a price one day.

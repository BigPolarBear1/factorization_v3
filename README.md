# factorization_v3

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WORK IN PROGRESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Useage: python3 QSv3.py -keysize 40

To do: Will only factor below 50 bits for now, THIS IS AS EXPECTED.
This is a first draft. I will be uploading improvements frequently now.
The approach here is to bypass smooth finding and directly take the GCD on quadratic coefficients.
Right now, it will bruteforce every coefficient and root combination, that is why its so slow.
I know the math behind it and will upload improvements soon.

In addition I also need to fix coefficient lifting to higher powers. That code is broken atm.

Once I finish with the main improvements I'll also upload an updated paper. I've written large sections of it already.

Hope to upload some of the big improvements I have in mind today or tomorrow.. these last few months have been shit, just really depressed. Like my head is working in slow motion.
It's annoying.. because I have been barely able to work this last year, but because of depression and being broke.. I also havn't had a chance to recover from it either.. if I hadn't been broke I would have gone backpacking for some months.
Anyway... almost there now.. everything should get uploaded within a week now... 

Update: I'll upload a fixed version tomorrow. Since we are doing calculations mod m, where m is constructed from mod p<sub>i</sub> and p is prime ... we need to adjust things to mod n. I know how to do it. And there's even a way to tell if a coefficient and root combination is even worth the cpu cycles to try and adjust to mod n. That's how I'm going to crack this wide open.... stay tuned :), going to be some good shit soon.
The thing thats broken right now is that before you take the derivative of the root and coefficient to calculate the other coefficient, you need to adjust the root so we are working mod n. Which is not super hard, its just understanding quadratic sequences.

I hate this place. I miss the friends I used to have but havnt seen in over a year bc of msft. For a few years, it was nice to learn to be myself. Here its back to my repressed self. I'm going to kill myself soon if this doesn't work. This isnt a life worth living anymore. Its just a pure nightmare now.

update: EUREKA!!!!! I GOT IT. PREPARE FOR YOUR WORST NIGHTMARE TOMORRROW, HAHAHAHAHAHAHAHAHAHAHHAHAA. I 100% got it now. This was the right direction. I finally found it! Tomorrow. Watch me. Its going to happen now.
Those roots of the quadratic... that was the secret. I shouldn't have zeroed in on the coefficients while ignoring the roots for 2 years. There's this really neat trick which easily yields the factorization of N. You'll see tomorrow.
Its a truely great reduction in complexity compared to quadratic sieve and v2.. what I found just now, no matter what, it will come online tomorrow... there's only one way forward. I must succeed. Even if it ruins my life, I do this for all the other people like me, so they dont need to get harassed and insulted for who they are. I want a better world. This world is fucking shit.

Update: Hope you are ready for today... it all finally fell into place now. I don't care. All that was good in my life has been destroyed by vile people, now I'll destroy them. Go to hell. Can't stop me. People don't understand what they are up against.

Its just like number field sieve!!! I understand it now how it works with these roots. I finally see it! PoC coming very soon.

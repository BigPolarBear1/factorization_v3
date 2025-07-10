I just uploaded QSv3_032.py<br />
Use: python3 QSv3_032.py -keysize 50

A work in progress.<br /><br />
To do:<br /><br />
-Fix lifting for powers of 2.<br />
-Get rid of trial division for smooth candidates... now that we can determine the exponent, this is redundant<br />
-Work out the details of a strategy to combine coefficients such that the factors outside the factor base shrink. <br />
This should be possible. I have a general idea in my head already on how to approach this now algorithmically. <br /><br />
Once all that is done, it should blast past the performance of factorization_v2.<br />
Then I need to write the paper, and start porting to c++ and optimize everything (and in addition use thing such as block lanczos instead of gaussian elimination)<br /><br />


Microsoft shouldn't have fired my manager. They will pay the price. I'm never stopping. I'll destroy the world if I must.

Update: Alright, I have come up with an outline for combining coefficients such that the number that is left when a smooth candidate is divided by the factor base becomes smaller, until it resolves into a smooth. Being able to build a lookup table for powers of primes is what enabled me to do this now. I'll go for a run, then slowly begin coding it and with some luck I ca upload it tomorrow. Once that is done, I just need to fix lifting for powers of 2 in the PoC and we're good to go. Should be done before the end of the week now....
Ah man, I feel so depressed. After everything that happened in 2023 and the persistent unemployment, isolation and hopelessness since... never before have I felt more of an urge to never interact with humans again and withdraw into nature. I like nature... nature is unforgiving, but there's no malice in nature... malice is an uniquely human attribute. I doubt it is a coincidence that while my own life is collapsing, the entire world is doing the same... people have lost their minds. I know why this is happening, because for the first time in history, people can be influenced at a scale like never before. Human psychology is a lot more fragile then people assume. Like all those dumb right wingers spreading lies about transgender people.... they know what they are doing... it's their playbook. They bring chaos into this world on purpose, because they are fucking idiots with 0 intelligence. And I'll punch them in their stupid faces. Hahaha.


Started wearing rainbow colors while out on my run. I think in a way it is coping with everything that happened while I was in the US. Often times, I am hoping for someone to get in my face again like those fucking shitheads with their gun in Redmond, I want to fight, I want violence so badly. Say something bad about gay people or trans people to my face. I beg you. Give me an excuse. Pieces of shit. I will gladly do final battle. Hahahaa. Fuck you all.

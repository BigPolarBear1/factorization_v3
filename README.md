Just uploaded QSv3_010.py

Useage: python3 QSv3_010.py -keysize 40

It is very slow still. It is as intended. Just uploading my work in progress for the day.
However this demonstrates indexing a hashmap by relative i-value. So that any coefficient pairings we find are garantueed to be at the correct i-value.

TO DO: I've added to the paper some info about how quadratic coefficients relate to the amount of times N divides the difference between both squares.
I feel like that was the last big thing I had to figure out.
All that's left now, is finding a fast algorithm leveraging all this knowledge.
Having a bit of trouble focusing. I looked deeper into number field sieve this weekend... but honestly, it is so wildly different from my own approach, its hard to find anything useful there.
Which is also a good thing, because I am starting to realize now, how truely different my own approach is to anything else out there.
But just improving standard quadratic sieve for slightly faster smooth finding (factorization_v2) isn't good enough. Lets see if I can pull of something better with all I know now in the coming days.....
the paper is also still a work in progress.... until I figure it out..

Update: Currently just messing around with taking the norm of a quadratic. I think this is the very last thing I need to understand deeper. There especially seems to be some interesting properties when we take the norm of both the leftside and rightside quadratic. I wonder if I can then pivot to something similar like number field sieve hmm, maybe not exactly the same, but similar.

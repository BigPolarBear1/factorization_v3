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

Hmm, I may be able to use the norm to solve this. So hopefully next week I can finally conclude my work..... we'll see. I started on factorization in June of 2023, so I guess I am now entering my third years of research. People must think I've lost my sanity... but the thing is... the more I learn, the more I understand, the more I am convinced I am on the right path... plus, looking at existing algorithms, I am also convinced that what I'm doing is unique and potentially better.. how can I stop knowing these things? In addition I have no future in 0days. That's a highly politicized industry... they dont want people like me selling 0days. Just a bunch of tech bros roleplaying surveillance shit.. kind of hate these people anyway, would 100% punch these fuckers in their face.

Update: Ergh, I just hate number field sieve. I see how it works, but everytime I go back to number field sieve to try and figure out how to solve my own work, I just realize that my own work is doing all these things already in a less confusing way. 

Update: Too warm today. Making some more progress though. Struggling really badly with depression. Havnt been able to see my friends in well over a year now ever since I flew back to Europe. I won't ever forigve microsoft for everything they have done. I'm not ever working for a company ever again. The corporate world is a vile place unless you're a tech bro. Rich assholes who always had it easy in life. I despise these people deeply. I despise them most of all for what they did to my former manager. One of the only person in this shit industry who ever believed in me only to get betrayed by the company he worked at for 27 years simply for being supportive of me. I swear, if I ever meet one of these people, I'll fucking end them. And that is no joke. I am dead serious.

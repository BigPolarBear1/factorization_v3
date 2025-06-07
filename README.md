Just uploaded v3_002:

useage: python QSv3_002.py -keysize 20

Only use 20 bit keys for this version. I've set all the parameters for 20 bit.
Its a proof of idea. Literally nothing more. To demonstrate the math from the paper.

Next I'm going to refactor the entire thing. Get rid of all the loops. 
There is a much better way to go about finding similar small coefficients mod p<sub>i</sub> .. but it is again a big refactoring project. Somewhere in the weekend it will get released.

Update: Instead of indexing that xN datastructure (or I<sub>i</sub>N in the paper) by means of xN, it is much easier to group small coefficients together from any xN mod p<sub>i</sub> and THEN calculate the corresponding xN value. 
Something like that.. hmm. I have a lot of ideas right now, but I am trying to come up with the MOST optimal one.. since I don't want to spent weeks doing rounds of refactoring my code. I want to get it right the first time around. I should go for a run in a few hours and think about this.

Update: Slept really poorly last night. So I don't know if I will get a lot of work done today. Just think about this world too much. "I have no mouth, but I must scream", this is often how I feel like. I have very poor verbal skills irl. People who have talked to me irl know this. I remember when I was living at this institute as a kid, one of the people working there one day would talk to me in a word document, because he seemed suprised I was a lot more well-spoken in written language. As an adult, this hasn't really changed. If I try really really well, and make sure to articulate clearly, I can hide it. But I think, people who don't know me, when they talk to me, they all think I have some mental disability. I guess, not having a driver's license and living with my parents at the age of 35 isn't helping my case. The thing I can't communicate, not even in written language, is the abstract thought happening in my head.. and the way I "abstractly" see and understand this world we live in, it causes so much pain and distress, but I am unable to communicate it. I really liked living in Vancouver and working together with my teamlead, for a moment, I could live with some dignitiy. Now I feel trapped again unable to communicate anything to anyone. I had hoped, maybe if I cant communicate as well in language, maybe I can give math a try.. but I'm not that great at it either.. I've spent 2 years on this one problem, and I am not at all happy with the result. I really wish I wasn't broke so I could go be in the arctic again. Just vast emptiness of nothing... for some reason, I feel at peace there... like a familiar place, where I've lived all my life... perhaps in a different life I was a polar bear.

I GOT IT!

Use: QSv3_007.py -keysize 50

Just added v007. Just some minor improvements over v6. 
As you see, when it finds a good relative i-value, we are almost garantueed to find a good coefficient (although we may need to add the modulus a couple of times).
The main bottleneck now is reducing the cartesian product of coefficients mod p<sub>i</sub>.
Which I know how to do... so expect it to be fixed shortly.
With that bottleneck removed soon, we'll see how fast it ends up being. After that we can gain some more performance here and there.. there is definitely a lot of i-values we can eliminate from the start which don't have valid solutions... so I may need to think how I'm going to do that in code... Lets see.
When all is said and done, I am hoping to overtake v2 in performance. Without having to find smooth numbers, which would be a major breakthrough. Plus all of this is in python... this has a lot of potential to be super-optimized since we only use straightforward operations.

Anyway, I will fix that Cartesian product issue tomorrow or friday. Going to relax for the rest of the day now because tomorrow I have to go to court for sending an angry email to the FBI. Since they seem hellbend on making a big deal out of it. Yea well, America ruined my life the way I see it. I had all my friends there. You people threaten me with a gun and kick me of the country, forcing me to leave behind the friends I had known for 4 years in the region (I should have stayed in Canada, moving to the US was one of the biggest regrets in my life, but I cant even go back to Canada because I was on a work visa). You people chose the path of cryptologic warfare, not me. A price will be paid. You can also thank Microsoft for firing my manager with blatant lies out of retaliation for his support of me, that, more then anything is what caused this.


Update: Can't sleep. If I am not active anymore after tomorrow, it means I have been robbed of my freedom. In which case, I hope someone will finish my work, it is basically done now... just need to address the size of that cartesian product...

Update: They just want me to do court ordered therapy for a year. Which is far less worse then what I had feared. Plus it also means the road is wide open for me now to finish my math research (short of getting drone striked by the Americans bc they are mad about my math.. which with the current administration woulldn't even be out of character). One final sprint now, nearly there. We're going to fix this world with math, hahahaha. 

Thinking about it, I don't think I have much future ambitions anymore. Because I just can't get over the destruction caused by microsoft anymore. It's almost 2 years ago, and I still feel mentally tormented by it. They really shouldn't have gone after my manager. Now I just want to finish my math. Something that screams "I WAS HERE!", and then dissappear into the arctic on some mad expedition. Be in the one place where I've always felt free and happy, far away from humans.


score = [0,0]
if any(score[:1]):  
    sample= 0
elif any(score[:4]):
    sample= 1
elif any(score[:16]):
    sample= 2
elif any(score[:64]):
    sample= 3
else:
    sample= 4
print(sample)
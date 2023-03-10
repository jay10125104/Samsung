import jieba.analyse
sentence = 'when someone commits a murder they typically go to extreme lengths to cover up their brutal crime . the harsh prison sentences that go along with killing someone are enough to deter most people from ever wanting to be caught , not to mention the intense social scrutiny they would face . occasionally , however , there are folks who come forward and admit guilt in their crime . this can be for any number of reasons , like to gain notoriety or to clear their conscience , though , in other instances , people do it to come clean to the people they care about . when rachel hutson was just 19 years old , she murdered her own mother in cold blood . as heinous and unimaginable as her crime was , it was what she did after that shocked people the most … rachel was just a teenager when she committed an unthinkable act against her own other … while that in and of itself was a heinous crime , it ’s what rachel did in the aftermath of her own mother ’s murder that shook people to their core . you ’re not going to believe what strange thing she decided to do next … it ’s hard to understand what drove rachel to commit this terrible act , but sending the photo afterward seems to make even less sense . share this heartbreaking story with your friends below .'
keywords = jieba.analyse.extract_tags(sentence, topK=20, withWeight=True)

for item in keywords:
    print(item[0], item[1])
print('*' * 42)
keywords = jieba.analyse.textrank(sentence, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns','eng'))
for item in keywords:
    print(item[0], item[1])
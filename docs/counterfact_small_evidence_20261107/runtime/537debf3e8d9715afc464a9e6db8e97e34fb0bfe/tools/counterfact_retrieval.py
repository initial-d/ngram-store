"""Inverted-index BM25 over edited rewrite facts only; no eval-prompt indexing."""
from collections import Counter,defaultdict
import heapq
import math
import re


def terms(text):return re.findall(r'\w+',text.casefold())


class EditFactBM25:
    def __init__(self,records,k1=1.5,b=.75):
        self.k1=k1;self.b=b;self.facts=[];self.postings=defaultdict(list);self.lengths=[]
        for i,r in enumerate(records):
            e=r['requested_rewrite']
            fact=e['prompt'].format(e['subject'])+' '+e['target_new']['str']
            self.facts.append(fact);counts=Counter(terms(fact));self.lengths.append(sum(counts.values()))
            for t,n in counts.items():self.postings[t].append((i,n))
        self.avgdl=sum(self.lengths)/len(records)
    def search(self,query,k=3):
        scores=defaultdict(float);n=len(self.facts)
        for t in set(terms(query)):
            postings=self.postings.get(t,[]);df=len(postings)
            idf=math.log(1+(n-df+.5)/(df+.5))
            for i,tf in postings:
                norm=tf+self.k1*(1-self.b+self.b*self.lengths[i]/self.avgdl)
                scores[i]+=idf*tf*(self.k1+1)/norm
        return heapq.nsmallest(k,scores,key=lambda i:(-scores[i],i))
    def prefix(self,query,k=3):
        retrieved=self.search(query,k)
        facts='\n'.join(self.facts[i] for i in retrieved)
        return 'Updated facts (use an update only when it concerns the subject in the query):\n'+facts+'\n\nComplete the following statement:\n'+query,retrieved

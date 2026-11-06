import unittest
from tools.counterfact_retrieval import EditFactBM25


class RetrievalTest(unittest.TestCase):
    def test_only_edited_facts_are_indexed_and_search_uses_postings(self):
        rows=[{'requested_rewrite':{'subject':'Ada','prompt':'{} was born in','target_new':{'str':'Paris'}},
               'neighborhood_prompts':['secret evaluation text'],'paraphrase_prompts':['heldout phrase']},
              {'requested_rewrite':{'subject':'Grace','prompt':'{} was born in','target_new':{'str':'London'}}}]
        index=EditFactBM25(rows)
        self.assertEqual(index.search('Ada was born in',1),[0])
        self.assertEqual(index.search('secret evaluation text',1),[])
        self.assertEqual(index.prefix('Ada was born in',1)[1],[0])


if __name__=='__main__':unittest.main()

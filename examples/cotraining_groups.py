"""Fixed-domain training/export recipe; correctness fixture, not research data.

Requires pip install 'ngram-store[torch]'. Original row packing stays separate.
"""
import tempfile
from pathlib import Path

import torch

from ngram_store import BagPartition, CertifiedBagRows, export_certified_bags, grouped_embedding_bag


def main():
    # In an application, derive bags from the frozen training vocabulary and
    # protect word/padding rows. Equal occurrence counts certify sharing.
    bags = torch.tensor([[0, 1, 2], [0, 3, 4]], dtype=torch.long)
    partition = BagPartition.fit([[1, 2], [3, 4]], 5, protected=[0])
    rows = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [2., 0.], [0., 2.]])
    model = grouped_embedding_bag(rows, partition, padding_idx=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    optimizer.zero_grad(set_to_none=True)
    model(bags).square().sum().backward()
    optimizer.step()
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / 'aggregate-store'
        export_certified_bags(model.expanded_rows().detach().numpy(), partition, output)
        with CertifiedBagRows(output) as native:
            plan = native.compile(bags.tolist(), padding_idx=0)
            actual = native.evaluate(plan)
            torch.testing.assert_close(torch.from_numpy(actual), model(bags))
            print({'partition': partition.summary(), 'query_plan': plan.summary()})
            try:
                native.aggregate([[1]])
            except ValueError as error:
                print({'partial_group_rejected': str(error)})
                print({'uniform_safety_diagnostic': partition.explain_query([1])})


if __name__ == '__main__':
    main()

from datascience_learning import datasets


def test_list_datasets():
    assert len(datasets.fake_datasets()) > 0


def test_check_datasets():
    dataset_list = datasets.fake_datasets()

    for d in dataset_list:
        x, y = datasets.fake_datasets(d)
        assert x.shape[0] == y.shape[0]

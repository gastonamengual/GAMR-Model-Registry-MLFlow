from app.model.input_data import InputData


def test_input_data_valid() -> None:
    data = InputData(X=[[1.0, 2.0]], y=[1, 0])
    assert data.X == [[1.0, 2.0]]
    assert data.y == [1, 0]


def test_input_data_no_y() -> None:
    data = InputData(X=[[1.0, 2.0], [3.0, 4.0]])
    assert data.X == [[1.0, 2.0], [3.0, 4.0]]
    assert data.y is None
